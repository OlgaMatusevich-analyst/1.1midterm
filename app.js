// app.js
// + Feature Report rendering
// + Loss chart from training history (loss & val_loss)
// + Validation split & early stopping controls
// + Optional class weights block (commented) if захочешь включить

export class App {
  constructor({ tf, Chart, dl, model, ui }) {
    this.tf = tf; this.Chart = Chart; this.dl = dl; this.model = model; this.ui = ui;
    this.dataset = null; this.charts = { balance:null, overtime:null, corr:null, loss:null };
    this.lastPreds = null; this.history = null;

    ui.csvFile.addEventListener('change', () => this.#onCSV());
    ui.prepBtn.addEventListener('click', () => this.#prepare());
    ui.edaBtn.addEventListener('click', () => this.#runEDA());
    ui.buildBtn.addEventListener('click', () => this.#build());
    ui.trainBtn.addEventListener('click', () => this.#train());
    ui.evalBtn.addEventListener('click', () => this.#evaluate());
    ui.saveBtn.addEventListener('click', () => this.model.save());
    ui.loadBtn.addEventListener('click', () => this.model.load().then(()=>this.#toggleTrainButtons(true)));
    ui.resetBtn.addEventListener('click', () => this.#reset());
    ui.downloadBtn.addEventListener('click', () => this.#downloadCSV());
    ui.thrAuto.addEventListener('click', async () => this.#autoThreshold());
  }

  async #onCSV() {
    try {
      this.#status('dataStatus','Reading…','#fef3c7','#92400e');
      await this.dl.fromFile(this.ui.csvFile.files[0]);
      this.ui.prepBtn.disabled = false;
      this.ui.edaBtn.disabled = false;
      this.#status('dataStatus','Loaded','#dcfce7','#166534');
    } catch (e) { this.#status('dataStatus','Error','#fee2e2','#991b1b'); alert(e.message || String(e)); }
  }

  async #runEDA() {
    try {
      const { balance, topCorr, catRates } = this.dl.eda();
      this.ui.edaText.innerHTML =
        `Баланс классов — Yes: <b>${balance.positive}</b>, No: <b>${balance.negative}</b> (доля ${(balance.rate*100).toFixed(1)}%).<br/>
         Топ-числовые корреляции: <b>${topCorr.map(([k,v])=>`${k} (${v.toFixed(2)})`).join(', ')}</b>.`;

      this.#renderBar('balance', this.ui.chartClassBalance,
        ['No','Yes'], [balance.negative, balance.positive], 'Class Balance');

      if (catRates.OverTime) {
        this.#renderBar('overtime', this.ui.chartOvertime,
          catRates.OverTime.map(d=>d.k),
          catRates.OverTime.map(d=>+(d.rate*100).toFixed(2)),
          'Attrition rate by OverTime (%)');
      }

      const corr = topCorr;
      this.#renderBar('corr', this.ui.chartCorr,
        corr.map(d=>d[0]), corr.map(d=>+(d[1]).toFixed(3)),
        'Top numeric correlations (Pearson)');
    } catch (e) { alert(e.message || String(e)); }
  }

  async #prepare() {
  try {
    this.#progress(0);

    // поддержка запятой в локали
    const splitNum = (typeof this.ui.testSplit.value === 'string'
      ? this.ui.testSplit.value.replace(',', '.')
      : this.ui.testSplit.value);
    const testSplit = Number(splitNum) / 100 || 0.2;

    // dispose prev
    this.dataset?.xTrain?.dispose?.(); this.dataset?.yTrain?.dispose?.();
    this.dataset?.xTest?.dispose?.();  this.dataset?.yTest?.dispose?.();

    // БЕЗ augment:
    this.dataset = this.dl.prepareTensors({ testSplit });

    // далее — как было (expandDims, отчёт фич и т.п.)
  } catch (e) { alert(e.message || String(e)); }
}



      // expand for GRU
      const xTr3 = this.dataset.xTrain.expandDims(1);
      const xTe3 = this.dataset.xTest.expandDims(1);
      this.dataset.xTrain.dispose(); this.dataset.xTest.dispose();
      this.dataset.xTrain = xTr3; this.dataset.xTest = xTe3;

      this.ui.buildBtn.disabled = false;

      // Feature report to UI
      const rep = this.dl.featureReport();
      const mk = (title, arr) => `<tr><th>${title}</th><td>${arr.length}</td><td class="mono">${arr.join(', ')}</td></tr>`;
      const desc = Object.entries(rep.createdDescriptions).map(([k,v])=>`<div><b>${k}</b>: ${v}</div>`).join('');
      this.ui.featReport.innerHTML = `
        <table>
          <thead><tr><th>Group</th><th>#</th><th>Attributes</th></tr></thead>
          <tbody>
            ${mk('Kept (inputs)', rep.kept)}
            ${mk('Dropped', rep.dropped)}
            ${mk('Created (engineered)', rep.created)}
          </tbody>
        </table>
        <div style="margin-top:8px">${desc}</div>
      `;
    } catch (e) { alert(e.message || String(e)); }
  }

  #build() {
    try {
      const timesteps = this.dataset?.xTrain?.shape?.[1];
      const features  = this.dataset?.xTrain?.shape?.[2];
      if (!timesteps || !features) throw new Error('Prepare dataset first.');

      const units = Math.max(8, Number(this.ui.units.value) | 0);
      const layers = Math.max(1, Number(this.ui.layers.value) | 0);
      const lr = Number(this.ui.lr.value) || 1e-3;

      this.model.build({ timesteps, features, units, layers, lr });
      this.#toggleTrainButtons(true);
    } catch (e) { alert(e.message || String(e)); }
  }

  async #train() {
    try {
      this.#progress(0);
      const epochs = Math.max(1, Number(this.ui.epochs.value) | 0);
      const batchSize = Math.max(1, Number(this.ui.batchSize.value) | 0);
      const validationSplit = Math.min(Math.max(Number(this.ui.valSplit.value) || 0.2, 0.05), 0.4);
      const patience = Math.max(2, Number(this.ui.patience.value) | 0);

      // OPTIONAL: class weights (закомментируй если не нужно)
      // const yArr = await this.dataset.yTrain.array();
      // const pos = yArr.filter(r => r[0] === 1).length;
      // const neg = yArr.length - pos;
      // const w1 = neg / Math.max(1, pos), w0 = 1;
      // const sampleWeight = tf.tensor1d(yArr.map(r => r[0] === 1 ? w1 : w0));

      this.history = await this.model.fit({
        xTrain: this.dataset.xTrain,
        yTrain: this.dataset.yTrain,
        epochs, batchSize, validationSplit, patience,
        // sampleWeight,
        onEpoch: (epoch, logs) => this.#progress((epoch+1)/epochs)
      });

      // sampleWeight?.dispose();
      this.ui.evalBtn.disabled = false;
      this.#progress(1);

      // Draw loss chart
      this.#drawLoss();
    } catch (e) { alert(e.message || String(e)); }
  }

  async #evaluate() {
    try {
      const thr = Number(this.ui.thr.value) || 0.5;
      const res = await this.model.evaluate({
        xTest: this.dataset.xTest, yTest: this.dataset.yTest, threshold: thr
      });
      this.#renderMetrics(res);
      this.lastPreds = await this.#collectPredictions(thr);
      this.ui.downloadBtn.disabled = false;
    } catch (e) { alert(e.message || String(e)); }
  }

  async #autoThreshold() {
    if (!this.dataset) return alert('Prepare & Train first.');
    const probs = this.model.predict(this.dataset.xTest);
    const p = await probs.array(); probs.dispose();
    const y = await this.dataset.yTest.array();

    let best = { thr: 0.5, f1: 0 };
    for (let t = 0.10; t <= 0.90; t += 0.01) {
      let tp=0, fp=0, fn=0;
      for (let i=0;i<p.length;i++){
        const pred = p[i][0] >= t ? 1 : 0;
        const gt = y[i][0];
        if (pred===1 && gt===1) tp++;
        else if (pred===1 && gt===0) fp++;
        else if (pred===0 && gt===1) fn++;
      }
      const prec = tp / Math.max(1, tp+fp);
      const rec  = tp / Math.max(1, tp+fn);
      const f1   = 2*prec*rec/Math.max(1e-9,prec+rec);
      if (f1 > best.f1) best = { thr: +t.toFixed(2), f1 };
    }
    this.ui.thr.value = best.thr.toFixed(2);
    alert(`Best F1 at threshold ${best.thr}: ${best.f1.toFixed(3)}`);
  }

  async #collectPredictions(threshold) {
    const probs = this.model.predict(this.dataset.xTest);
    const pArr = await probs.array(); probs.dispose();
    const yArr = await this.dataset.yTest.array();
    const meta = this.dataset.testMeta || [];
    const out = [];
    for (let i=0;i<pArr.length;i++){
      const prob = pArr[i][0];
      const pred = prob >= threshold ? 'Yes' : 'No';
      const truth = yArr[i][0] === 1 ? 'Yes' : 'No';
      const m = meta[i] || {};
      out.push({
        EmployeeNumber: m.EmployeeNumber ?? '',
        JobRole: m.JobRole ?? '',
        OverTime: m.OverTime ?? '',
        YearsAtCompany: m.YearsAtCompany ?? '',
        MonthlyIncome: m.MonthlyIncome ?? '',
        Probability: +prob.toFixed(6),
        Predicted: pred, True: truth
      });
    }
    return out;
  }

  #downloadCSV() {
    if (!this.lastPreds?.length) return alert('No predictions to download. Run Evaluate first.');
    const cols = ['EmployeeNumber','JobRole','OverTime','YearsAtCompany','MonthlyIncome','Probability','Predicted','True'];
    const header = cols.join(',');
    const rows = this.lastPreds.map(r => cols.map(c => `${String(r[c] ?? '').replace(/,/g,';')}`).join(','));
    const csv = [header, ...rows].join('\n');
    const blob = new Blob([csv], {type:'text/csv'}); const url = URL.createObjectURL(blob);
    const a = document.createElement('a'); const ts = new Date().toISOString().slice(0,19).replace(/[:T]/g,'-');
    a.href = url; a.download = `attrition_predictions_${ts}.csv`; a.click(); URL.revokeObjectURL(url);
  }

 #renderMetrics({ prec, rec, f1, auc, cm }) {
  const fmt = v => Number.isFinite(v) ? v.toFixed(4) : '–';

  // accuracy из матрицы ошибок
  const total = (cm.tp ?? 0) + (cm.tn ?? 0) + (cm.fp ?? 0) + (cm.fn ?? 0);
  const acc = total ? ((cm.tp ?? 0) + (cm.tn ?? 0)) / total : NaN;

  // метрики
  this.ui.elPrec.textContent = fmt(prec);
  this.ui.elRec.textContent  = fmt(rec);
  this.ui.elF1.textContent   = fmt(f1);
  if (this.ui.elAcc) this.ui.elAcc.textContent = fmt(acc);  // <-- новая строка
  this.ui.elAUC.textContent  = fmt(auc);

  // матрица ошибок
  this.ui.cmTN.textContent = cm.tn ?? '–';
  this.ui.cmFP.textContent = cm.fp ?? '–';
  this.ui.cmFN.textContent = cm.fn ?? '–';
  this.ui.cmTP.textContent = cm.tp ?? '–';
}


  #drawLoss() {
    if (!this.history) return;
    const labels = this.history.loss.map((_,i)=>`${i+1}`);
    const ds = [{label:'loss', data:this.history.loss}];
    if (this.history.val_loss?.length) ds.push({label:'val_loss', data:this.history.val_loss});
    const ctx = this.ui.lossChart.getContext('2d');
    this.charts.loss?.destroy?.();
    this.charts.loss = new this.Chart(ctx, {
      type: 'line',
      data: { labels, datasets: ds },
      options: { responsive:true, plugins:{ legend:{ position:'bottom' } }, interaction:{ mode:'index', intersect:false } }
    });
    const last = this.history.loss.at(-1);
    const lastV = this.history.val_loss?.at(-1);
    this.ui.trainSummary.innerHTML = `Последняя loss: <b>${last?.toFixed(4)}</b>${Number.isFinite(lastV)?`, val_loss: <b>${lastV.toFixed(4)}</b>`:''}`;
  }

  #renderBar(key, canvas, labels, data, title) {
    const ctx = canvas.getContext('2d');
    this.charts[key]?.destroy?.();
    this.charts[key] = new this.Chart(ctx, {
      type: 'bar',
      data: { labels, datasets: [{ label: title, data }] },
      options: { responsive: true, plugins: { legend: { display:false }, title: { display:true, text:title } } }
    });
  }

  #toggleTrainButtons(enable) { this.ui.buildBtn.disabled = !enable; this.ui.trainBtn.disabled = !enable; this.ui.evalBtn.disabled = true; this.ui.saveBtn.disabled = !enable; this.ui.downloadBtn.disabled = true; }
  #status(id,text,bg,color){ const n=document.getElementById(id); if(n){ n.textContent=text; n.style.background=bg; n.style.color=color; } }
  #progress(v){ this.ui.prog.value = Math.max(0, Math.min(1, v)); }
  #reset() {
    try {
      this.model.dispose();
      if (this.dataset) { this.dataset.xTrain.dispose(); this.dataset.yTrain.dispose(); this.dataset.xTest.dispose(); this.dataset.yTest.dispose(); }
      this.dataset = null; this.lastPreds = null; this.history = null;
      this.ui.prepBtn.disabled = true; this.#toggleTrainButtons(false); this.#progress(0);
      this.ui.trainSummary.textContent = ''; this.ui.featReport.innerHTML = '';
      this.#renderMetrics({prec:NaN, rec:NaN, f1:NaN, auc:NaN, cm:{tp:'–',tn:'–',fp:'–',fn:'–'}});
      this.charts.loss?.destroy?.();
    } catch(_e) {}
  }
}
