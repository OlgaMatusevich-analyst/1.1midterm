// data-loader.js
// Parse local CSV, EDA, one-hot for categoricals, standardize numerics, train/test split.
// Removes noisy/constant features from model input: EmployeeNumber, StandardHours, EmployeeCount, Over18 (как фича).
// Добавлена аугментация позитивного класса (после стандартизации train): target ratio + шум по числовым фичам.

export class DataLoader {
  constructor(opts = {}) {
    this.log = opts.log || (() => {});
    this.headers = [];
    this.raw = [];
    this.labelKey = 'Attrition';
    this.attritionMap = { Yes: 1, No: 0 };

    // Числовые (очищено от шумных/константных)
    this.numCols = new Set([
      'Age','DailyRate','DistanceFromHome','Education',
      'EnvironmentSatisfaction','HourlyRate','JobInvolvement','JobLevel','JobSatisfaction',
      'MonthlyIncome','MonthlyRate','NumCompaniesWorked','PercentSalaryHike','PerformanceRating',
      'RelationshipSatisfaction','StockOptionLevel','TotalWorkingYears',
      'TrainingTimesLastYear','WorkLifeBalance','YearsAtCompany','YearsInCurrentRole',
      'YearsSinceLastPromotion','YearsWithCurrManager'
    ]);

    // Категориальные (без Over18)
    this.catKnown = new Set([
      'BusinessTravel','Department','EducationField','Gender','JobRole','MaritalStatus','OverTime'
    ]);

    this.catCols = [];
    this.encoders = {}; // { col: {value:index} }
    this.featureOrder = [];
    this.scaler = null; // { mean:[], std:[] }

    // Мета-поля для экспорта (не идут в модель)
    this.metaFields = ['EmployeeNumber','JobRole','OverTime','YearsAtCompany','MonthlyIncome'];
    this.metaRows = []; // по исходному индексу строки
  }

  async fromFile(file) {
    if (!file) throw new Error('No file selected.');
    const text = await this.#readFileText(file);
    const { headers, rows } = this.#parseCSV(text);

    if (!headers.includes(this.labelKey)) {
      throw new Error(`Missing required column: ${this.labelKey}`);
    }

    this.headers = headers;
    this.catCols = headers.filter(h => this.catKnown.has(h));
    const allowed = new Set([...this.numCols, ...this.catCols, this.labelKey, ...this.metaFields]);

    this.raw = rows.map(r => {
      const obj = {};
      for (const k of headers) if (allowed.has(k)) obj[k] = r[k];
      return obj;
    });

    // Кешируем мета
    this.metaRows = this.raw.map(r => {
      const m = {};
      for (const f of this.metaFields) m[f] = r[f] ?? '';
      return m;
    });

    this.log(`Loaded ${this.raw.length} rows (${this.catCols.length} categorical, ${this.numCols.size} numeric).`);
    return this;
  }

  // ================= EDA =================
  eda() {
    if (!this.raw?.length) throw new Error('Load data first.');
    const n = this.raw.length;

    // баланс классов
    let pos = 0, neg = 0;
    for (const r of this.raw) (String(r[this.labelKey]) === 'Yes') ? pos++ : neg++;
    const balance = { positive: pos, negative: neg, rate: pos / Math.max(1, n) };

    // корреляции числовых с Attrition (0/1)
    const y = this.raw.map(r => (String(r[this.labelKey]) === 'Yes' ? 1 : 0));
    const corr = {};
    for (const col of this.numCols) {
      const x = this.raw.map(r => Number(r[col] ?? 0));
      corr[col] = this.#pearson(x, y);
    }
    const topCorr = Object.entries(corr).sort((a,b)=>Math.abs(b[1])-Math.abs(a[1])).slice(0,8);

    // ставки увольнений по OverTime и JobRole
    const catRates = {};
    for (const col of ['OverTime','JobRole']) {
      if (!this.catCols.includes(col) && col !== 'JobRole') continue;
      const rates = {};
      for (const r of this.raw) {
        const k = r[col] ?? '';
        if (!rates[k]) rates[k] = { pos:0, total:0 };
        rates[k].pos += (String(r[this.labelKey])==='Yes') ? 1 : 0;
        rates[k].total += 1;
      }
      const out = Object.entries(rates).map(([k,v]) => ({ k, rate: v.pos/Math.max(1,v.total) }))
                                      .sort((a,b)=>b.rate-a.rate);
      catRates[col] = out;
    }

    return { balance, topCorr, catRates };
  }

  // ============== Tensors & Splits ==============
  prepareTensors({ testSplit = 0.2, augment = null }) {
    if (!this.raw?.length) throw new Error('Dataset not loaded.');
    this._augCfg = augment || { enable:false };

    // кодировщики для категорий
    this.#fitCategoricals(this.raw);

    // признаки + целевая
    const feats = [];
    const labels = [];
    for (const r of this.raw) {
      const { v, y } = this.#rowToFeatures(r);
      feats.push(v); labels.push(y);
    }

    // === STRATIFIED split: сохраняем долю "Yes" в train/test ===
const posIdx = [], negIdx = [];
for (let i = 0; i < feats.length; i++) {
  (labels[i] === 1 ? posIdx : negIdx).push(i);
}

// детерминированно перемешаем каждую группу (фиксированный seed)
const shuffle = (arr) => {
  let seed = 1337;
  const rand = () => (seed = (seed*1664525 + 1013904223) % 2**32) / 2**32;
  for (let i = arr.length - 1; i > 0; i--) {
    const j = Math.floor(rand() * (i + 1));
    [arr[i], arr[j]] = [arr[j], arr[i]];
  }
  return arr;
};
shuffle(posIdx); shuffle(negIdx);

// сколько всего уходит в test
const nTest = Math.max(1, Math.floor(feats.length * Math.min(Math.max(testSplit, 0.05), 0.9)));

// сколько позитивов держать в тесте (пропорционально общей доле)
let testPos = Math.round(nTest * (posIdx.length / Math.max(1, feats.length)));
testPos = Math.min(testPos, posIdx.length);
testPos = Math.max(0, Math.min(testPos, nTest));

// убедимся, что негативов хватит закрыть остаток
let testNeg = nTest - testPos;
if (testNeg > negIdx.length) {
  testNeg = negIdx.length;
  testPos = nTest - testNeg;
}

// финальные индексы
const testIdx  = posIdx.slice(0, testPos).concat(negIdx.slice(0, testNeg));
const trainIdx = posIdx.slice(testPos).concat(negIdx.slice(testNeg));


    let Xtr = trainIdx.map(i => feats[i]);
    let ytr = trainIdx.map(i => [labels[i]]);
    const Xte = testIdx.map(i => feats[i]);
    const yte = testIdx.map(i => [labels[i]]);

    // стандартизация по train
    this.#fitScaler(Xtr);
    let XtrS = this.#applyScaler(Xtr);
    const XteS = this.#applyScaler(Xte);

    // --- NEW: аугментация позитивов на train (после стандартизации) ---
    if (this._augCfg.enable) {
      const numDim = this.numCols.size; // числовые идут первыми
      const { X, Y } = this.#augmentPositivesOnScaled(
        XtrS, ytr,
        numDim,
        Math.min(Math.max(Number(this._augCfg.targetRatio) || 0.5, 0.2), 0.8),
        Math.min(Math.max(Number(this._augCfg.noiseStd) || 0.05, 0.0), 0.2)
      );
      XtrS = X; ytr = Y;
      this.log(`Augmented positives to target ratio ${this._augCfg.targetRatio}; train rows: ${XtrS.length}`);
    }

    const xTrain = tf.tensor2d(XtrS);
    const yTrain = tf.tensor2d(ytr);
    const xTest  = tf.tensor2d(XteS);
    const yTest  = tf.tensor2d(yte);

    // мета для теста (экспорт)
    const testMeta = testIdx.map(i => this.metaRows[i]);

    this.log(`Prepared tensors: xTrain ${xTrain.shape}, yTrain ${yTrain.shape}, xTest ${xTest.shape}, yTest ${yTest.shape}`);
    return {
      xTrain, yTrain, xTest, yTest,
      testMeta,
      attritionMap: this.attritionMap,
      featureOrder: this.featureOrder.slice()
    };
  }

  // ------------- helpers -------------
  #readFileText(file) {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onerror = () => reject(new Error('Failed to read file.'));
      reader.onload = () => resolve(String(reader.result));
      reader.readAsText(file);
    });
  }

  #parseCSV(text) {
    const lines = text.replace(/\r\n/g, '\n').replace(/\r/g, '\n').split('\n').filter(Boolean);
    if (lines.length < 2) throw new Error('CSV has no data.');
    const headers = lines[0].split(',').map(h => h.trim());
    const rows = [];
    for (let i = 1; i < lines.length; i++) {
      const parts = lines[i].split(',');
      if (parts.length !== headers.length) continue;
      const obj = {};
      for (let j = 0; j < headers.length; j++) obj[headers[j]] = parts[j].trim();
      rows.push(obj);
    }
    return { headers, rows };
  }

  #fitCategoricals(rows) {
    this.encoders = {};
    for (const c of this.catCols) {
      const set = new Set();
      for (const r of rows) set.add(r[c] ?? '');
      const cats = Array.from(set.values()).sort();
      const enc = {}; cats.forEach((v,i)=>enc[v]=i);
      this.encoders[c] = enc;
    }
    const order = [];
    for (const col of this.numCols) order.push(col);
    for (const c of this.catCols) {
      const enc = this.encoders[c];
      for (const v of Object.keys(enc)) order.push(`${c}__${v}`);
    }
    this.featureOrder = order;
  }

  #rowToFeatures(r) {
    const feat = [];
    for (const col of this.numCols) {
      const v = Number(r[col] ?? 0);
      feat.push(Number.isFinite(v) ? v : 0);
    }
    for (const c of this.catCols) {
      const enc = this.encoders[c];
      const vec = new Array(Object.keys(enc).length).fill(0);
      const idx = enc[r[c] ?? ''];
      if (Number.isInteger(idx)) vec[idx] = 1;
      feat.push(...vec);
    }
    const y = this.attritionMap[String(r[this.labelKey] || 'No')] ?? 0;
    return { v: feat, y };
  }

  #fitScaler(X) {
    const d = X[0]?.length || 0;
    const mean = new Array(d).fill(0);
    const std  = new Array(d).fill(0);
    const n = X.length || 1;

    for (let j=0;j<d;j++){
      let s=0,s2=0;
      for (let i=0;i<n;i++){ const v=X[i][j]; s+=v; s2+=v*v; }
      mean[j]=s/n; std[j]=Math.sqrt(Math.max(1e-9, s2/n - mean[j]*mean[j]));
    }
    this.scaler = { mean, std };
  }
  #applyScaler(X) {
    const { mean,std } = this.scaler;
    return X.map(row => row.map((v,j)=>(v-mean[j])/std[j]));
  }

  #deterministicShuffleIndex(n) {
    const a = Array.from({length:n}, (_,i)=>i);
    let seed = 1337;
    const rand = () => (seed = (seed*1664525 + 1013904223) % 2**32) / 2**32;
    for (let i=n-1;i>0;i--) { const j = Math.floor(rand()*(i+1)); [a[i],a[j]]=[a[j],a[i]]; }
    return a;
  }

  #pearson(x, y) {
    const n = Math.min(x.length, y.length);
    let sx=0, sy=0, sxx=0, syy=0, sxy=0;
    for (let i=0;i<n;i++){ const xi=+x[i]; const yi=+y[i];
      sx+=xi; sy+=yi; sxx+=xi*xi; syy+=yi*yi; sxy+=xi*yi; }
    const cov = sxy/n - (sx/n)*(sy/n);
    const vx = sxx/n - (sx/n)*(sx/n);
    const vy = syy/n - (sy/n)*(sy/n);
    const denom = Math.sqrt(Math.max(vx*vy,1e-12));
    return cov/denom;
  }

  // --- NEW: аугментация позитивного класса на стандартизованных TRAIN-фичах ---
  #augmentPositivesOnScaled(XtrScaled, ytr, numDim, targetRatio=0.5, noiseStd=0.05) {
    const posIdx = [], negIdx = [];
    for (let i=0;i<ytr.length;i++) (ytr[i][0]===1 ? posIdx : negIdx).push(i);

    const P = posIdx.length, N = negIdx.length;
    if (P === 0) return { X: XtrScaled, Y: ytr };

    // решаем уравнение: P* / (P* + N) = targetRatio  → P* = targetRatio * (P* + N)
    const wantPos = Math.floor((targetRatio * N) / (1 - targetRatio)); // без учёта исходных P
    const need = Math.max(0, wantPos - P);
    if (need <= 0) return { X: XtrScaled, Y: ytr };

    const outX = XtrScaled.slice(), outY = ytr.slice();

    // нормальный шум (Box-Muller)
    const gauss = () => {
      let u=0, v=0; while(u===0) u=Math.random(); while(v===0) v=Math.random();
      return Math.sqrt(-2*Math.log(u))*Math.cos(2*Math.PI*v);
    };

    for (let k=0; k<need; k++) {
      const j = posIdx[k % P];
      const base = XtrScaled[j];
      const clone = base.slice();
      for (let d=0; d<numDim; d++) clone[d] = base[d] + gauss()*noiseStd; // шевелим только числа
      outX.push(clone);
      outY.push([1]);
    }
    return { X: outX, Y: outY };
  }
}
