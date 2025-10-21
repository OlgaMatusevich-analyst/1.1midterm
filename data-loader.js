// data-loader.js
// + Feature Engineering report: created/kept/dropped attributes are shown in UI.
// + Engineered features (numeric):
//   1) TenureRatio = YearsAtCompany / max(1, TotalWorkingYears)
//   2) NoPromotionRatio = YearsSinceLastPromotion / max(1, YearsAtCompany)
//   3) IncomePerLevel = MonthlyIncome / max(1, JobLevel)
//   4) OvertimeStress = (OverTime=='Yes') * (4 - JobSatisfaction)
//   5) TravelFreq = (BusinessTravel=='Travel_Frequently') ? 1 : 0
//   6) IsSingle = (MaritalStatus=='Single') ? 1 : 0
//   7) LongDistance = DistanceFromHome > 20 (threshold)
// Removed as noisy/constant from inputs: EmployeeNumber, EmployeeCount, StandardHours, Over18.

export class DataLoader {
  constructor(opts = {}) {
    this.log = opts.log || (() => {});
    this.headers = [];
    this.raw = [];
    this.labelKey = 'Attrition';
    this.attritionMap = { Yes: 1, No: 0 };

    // numeric base (cleaned)
    this.baseNum = [
      'Age','DailyRate','DistanceFromHome','Education',
      'EnvironmentSatisfaction','HourlyRate','JobInvolvement','JobLevel','JobSatisfaction',
      'MonthlyIncome','MonthlyRate','NumCompaniesWorked','PercentSalaryHike','PerformanceRating',
      'RelationshipSatisfaction','StockOptionLevel','TotalWorkingYears',
      'TrainingTimesLastYear','WorkLifeBalance','YearsAtCompany','YearsInCurrentRole',
      'YearsSinceLastPromotion','YearsWithCurrManager'
    ];

    this.dropped = ['EmployeeNumber','EmployeeCount','StandardHours','Over18']; // removed from inputs
    this.catKnown = ['BusinessTravel','Department','EducationField','Gender','JobRole','MaritalStatus','OverTime'];

    // engineered spec: array of {name, desc}
    this.engineeredSpec = [
      { name:'TenureRatio',       desc:'YearsAtCompany / max(1, TotalWorkingYears)' },
      { name:'NoPromotionRatio',  desc:'YearsSinceLastPromotion / max(1, YearsAtCompany)' },
      { name:'IncomePerLevel',    desc:'MonthlyIncome / max(1, JobLevel)' },
      { name:'OvertimeStress',    desc:"(OverTime=='Yes') * (4 - JobSatisfaction)" },
      { name:'TravelFreq',        desc:"BusinessTravel=='Travel_Frequently' ? 1 : 0" },
      { name:'IsSingle',          desc:"MaritalStatus=='Single' ? 1 : 0" },
      { name:'LongDistance',      desc:'DistanceFromHome > 20 ? 1 : 0' }
    ];

    this.catCols = [];
    this.encoders = {};         // { col: {value:index} } for real categoricals (not engineered dummies)
    this.featureOrder = [];     // numeric + engineered + one-hot categoricals
    this.scaler = null;         // { mean:[], std:[] }
    this.metaFields = ['EmployeeNumber','JobRole','OverTime','YearsAtCompany','MonthlyIncome'];
    this.metaRows = [];

    // computed lists for report
    this.kept = [];     // final numeric names + one-hot names (report prints high-level)
    this.created = this.engineeredSpec.map(x=>x.name);
  }

  async fromFile(file) {
    if (!file) throw new Error('No file selected.');
    const text = await this.#readFileText(file);
    const { headers, rows } = this.#parseCSV(text);

    if (!headers.includes(this.labelKey)) throw new Error(`Missing required column: ${this.labelKey}`);

    this.headers = headers;
    this.catCols = headers.filter(h => this.catKnown.includes(h));
    const allowed = new Set([...this.baseNum, ...this.catCols, this.labelKey, ...this.metaFields, ...this.dropped]);

    this.raw = rows.map(r => {
      const obj = {};
      for (const k of headers) if (allowed.has(k)) obj[k] = r[k];
      return obj;
    });

    this.metaRows = this.raw.map(r => {
      const m = {};
      for (const f of this.metaFields) m[f] = r[f] ?? '';
      return m;
    });

    this.log(`Loaded ${this.raw.length} rows (categorical=${this.catCols.length}, numeric=${this.baseNum.length}, dropped=${this.dropped.length}).`);
    return this;
  }

  // ---------- EDA ----------
  eda() {
    if (!this.raw?.length) throw new Error('Load data first.');
    const n = this.raw.length;

    let pos = 0, neg = 0;
    for (const r of this.raw) (String(r[this.labelKey]) === 'Yes') ? pos++ : neg++;
    const balance = { positive: pos, negative: neg, rate: pos / Math.max(1, n) };

    const y = this.raw.map(r => (String(r[this.labelKey]) === 'Yes' ? 1 : 0));
    const corr = {};
    for (const col of this.baseNum) {
      const x = this.raw.map(r => Number(r[col] ?? 0));
      corr[col] = this.#pearson(x, y);
    }
    const topCorr = Object.entries(corr).sort((a,b)=>Math.abs(b[1])-Math.abs(a[1])).slice(0,8);

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

  // ---------- Tensors & Split ----------
  prepareTensors({ testSplit = 0.2 }) {
  // никаких ссылок на augment / this._augCfg



    // Fit categorical encoders for *real* categoricals
    this.#fitCategoricals(this.raw);

    // Build feature order: base numerics + engineered + one-hot categoricals
    this.featureOrder = [];
    this.featureOrder.push(...this.baseNum);
    this.featureOrder.push(...this.engineeredSpec.map(f=>f.name));
    for (const c of this.catCols) {
      const enc = this.encoders[c];
      for (const v of Object.keys(enc)) this.featureOrder.push(`${c}__${v}`);
    }

    // feats + labels
    const feats = [];
    const labels = [];
    for (const r of this.raw) {
      const { v, y } = this.#rowToFeatures(r);
      feats.push(v); labels.push(y);
    }

    // stratified split
    const posIdx = [], negIdx = [];
    for (let i=0;i<feats.length;i++) (labels[i]===1 ? posIdx : negIdx).push(i);
    const shuffle = (arr) => { let seed=1337; const rand=()=> (seed=(seed*1664525+1013904223)%2**32)/2**32;
      for (let i=arr.length-1;i>0;i--){ const j=Math.floor(rand()*(i+1)); [arr[i],arr[j]]=[arr[j],arr[i]]; } return arr; };
    shuffle(posIdx); shuffle(negIdx);
    const nTest = Math.max(1, Math.floor(feats.length * Math.min(Math.max(testSplit,0.05),0.9)));
    let testPos = Math.round(nTest * (posIdx.length / Math.max(1, feats.length)));
    testPos = Math.max(0, Math.min(testPos, posIdx.length)); let testNeg = Math.min(nTest - testPos, negIdx.length);
    const testIdx  = posIdx.slice(0, testPos).concat(negIdx.slice(0, testNeg));
    const trainIdx = posIdx.slice(testPos).concat(negIdx.slice(testNeg));

    let Xtr = trainIdx.map(i => feats[i]);
    let ytr = trainIdx.map(i => [labels[i]]);
    const Xte = testIdx.map(i => feats[i]);
    const yte = testIdx.map(i => [labels[i]]);

    // scale on train
    this.#fitScaler(Xtr);
    let XtrS = this.#applyScaler(Xtr);
    const XteS = this.#applyScaler(Xte);

    // optional augmentation (after scaling)
    if (this._augCfg.enable) {
      const numDim = this.baseNum.length + this.engineeredSpec.length;
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

    const testMeta = testIdx.map(i => this.metaRows[i]);
    // Prepare simple high-level lists for report (not listing each one-hot)
    this.kept = [...this.baseNum, ...this.created, ...this.catCols];

    return { xTrain, yTrain, xTest, yTest, testMeta,
             attritionMap: this.attritionMap, featureOrder: this.featureOrder.slice() };
  }

  featureReport() {
    return {
      kept: this.kept.slice(),
      dropped: this.dropped.slice(),
      created: this.created.slice(),
      createdDescriptions: Object.fromEntries(this.engineeredSpec.map(x=>[x.name, x.desc]))
    };
  }

  // ---------- helpers ----------
  #readFileText(file) { return new Promise((res,rej)=>{ const r=new FileReader(); r.onerror=()=>rej(new Error('Failed to read file.')); r.onload=()=>res(String(r.result)); r.readAsText(file); }); }
  #parseCSV(text) { const lines=text.replace(/\r\n/g,'\n').replace(/\r/g,'\n').split('\n').filter(Boolean);
    if (lines.length<2) throw new Error('CSV has no data.');
    const headers=lines[0].split(',').map(h=>h.trim()); const rows=[];
    for (let i=1;i<lines.length;i++){ const parts=lines[i].split(','); if (parts.length!==headers.length) continue;
      const obj={}; for (let j=0;j<headers.length;j++) obj[headers[j]]=parts[j].trim(); rows.push(obj); }
    return { headers, rows }; }

  #fitCategoricals(rows) {
    this.encoders = {};
    for (const c of this.catCols) {
      const set = new Set(); for (const r of rows) set.add(r[c] ?? '');
      const cats = Array.from(set.values()).sort();
      const enc = {}; cats.forEach((v,i)=>enc[v]=i); this.encoders[c] = enc;
    }
  }

  #rowToFeatures(r) {
    const feat = [];
    // base numerics
    for (const col of this.baseNum) {
      const v = Number(r[col] ?? 0);
      feat.push(Number.isFinite(v) ? v : 0);
    }
    // engineered
    const YA = Math.max(1, Number(r['YearsAtCompany'] ?? 0));
    const TW = Math.max(1, Number(r['TotalWorkingYears'] ?? 0));
    const YP = Number(r['YearsSinceLastPromotion'] ?? 0);
    const JL = Math.max(1, Number(r['JobLevel'] ?? 1));
    const MI = Number(r['MonthlyIncome'] ?? 0);
    const JS = Number(r['JobSatisfaction'] ?? 0);
    const DF = Number(r['DistanceFromHome'] ?? 0);
    const OT = String(r['OverTime'] ?? '') === 'Yes' ? 1 : 0;
    const BT = String(r['BusinessTravel'] ?? '');
    const MS = String(r['MaritalStatus'] ?? '');

    const TenureRatio      = (Number(r['YearsAtCompany'] ?? 0)) / TW;
    const NoPromotionRatio = YP / YA;
    const IncomePerLevel   = MI / JL;
    const OvertimeStress   = OT * (4 - JS);
    const TravelFreq       = (BT === 'Travel_Frequently') ? 1 : 0;
    const IsSingle         = (MS === 'Single') ? 1 : 0;
    const LongDistance     = DF > 20 ? 1 : 0;

    feat.push(TenureRatio, NoPromotionRatio, IncomePerLevel, OvertimeStress, TravelFreq, IsSingle, LongDistance);

    // one-hot categoricals
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
    const d = X[0]?.length || 0; const mean=new Array(d).fill(0), std=new Array(d).fill(0); const n=X.length||1;
    for (let j=0;j<d;j++){ let s=0,s2=0; for (let i=0;i<n;i++){ const v=X[i][j]; s+=v; s2+=v*v; }
      mean[j]=s/n; std[j]=Math.sqrt(Math.max(1e-9, s2/n - mean[j]*mean[j])); }
    this.scaler = { mean, std };
  }
  #applyScaler(X) { const {mean,std}=this.scaler; return X.map(row=>row.map((v,j)=>(v-mean[j])/std[j])); }

  #pearson(x, y) {
    const n=Math.min(x.length,y.length); let sx=0,sy=0,sxx=0,syy=0,sxy=0;
    for (let i=0;i<n;i++){ const xi=+x[i]; const yi=+y[i]; sx+=xi; sy+=yi; sxx+=xi*xi; syy+=yi*yi; sxy+=xi*yi; }
    const cov = sxy/n - (sx/n)*(sy/n), vx=sxx/n - (sx/n)**2, vy=syy/n - (sy/n)**2;
    return cov/Math.sqrt(Math.max(vx*vy,1e-12));
  }

  #augmentPositivesOnScaled(Xtr, ytr, numDim, targetRatio=0.5, noiseStd=0.05) {
    const posIdx=[], negIdx=[]; for (let i=0;i<ytr.length;i++) (ytr[i][0]===1?posIdx:negIdx).push(i);
    const P=posIdx.length, N=negIdx.length; if (P===0) return {X:Xtr, Y:ytr};
    const wantPos = Math.floor((targetRatio * N) / (1 - targetRatio));
    const need = Math.max(0, wantPos - P); if (need<=0) return {X:Xtr, Y:ytr};
    const outX=Xtr.slice(), outY=ytr.slice();
    const gauss=()=>{ let u=0,v=0; while(u===0)u=Math.random(); while(v===0)v=Math.random(); return Math.sqrt(-2*Math.log(u))*Math.cos(2*Math.PI*v); };
    for (let k=0;k<need;k++){ const j=posIdx[k%P]; const base=Xtr[j]; const clone=base.slice();
      for (let d=0; d<numDim; d++) clone[d]=base[d]+gauss()*noiseStd;
      outX.push(clone); outY.push([1]); }
    return { X: outX, Y: outY };
  }
}
