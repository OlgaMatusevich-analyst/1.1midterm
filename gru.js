// gru.js
// + validationSplit support, + early stopping (patience), + history return for loss chart.

export class GRUClassifier {
  constructor(opts = {}) {
    this.log = opts.log || (() => {});
    this.model = null;
    this.inputShape = null;
  }

  build({ timesteps = 1, features, units = 128, layers = 1, lr = 1e-3 }) {
    if (!Number.isInteger(features) || features < 1) throw new Error(`Invalid features: ${features}`);
    if (!Number.isInteger(timesteps) || timesteps < 1) throw new Error(`Invalid timesteps: ${timesteps}`);

    this.dispose();
    this.inputShape = [timesteps, features];

    const m = tf.sequential();
    const U = Math.max(8, units|0), L = Math.max(1, layers|0);

    m.add(tf.layers.gru({
      units: U, inputShape: this.inputShape,
      returnSequences: L > 1, dropout: 0.2, kernelInitializer: 'heNormal'
    }));
    for (let i = 1; i < L; i++) {
      const isLast = i === L - 1;
      m.add(tf.layers.gru({ units: U, returnSequences: !isLast, dropout: 0.2, kernelInitializer: 'heNormal' }));
    }
    m.add(tf.layers.dense({ units: 64, activation: 'relu', kernelInitializer: 'heNormal' }));
    m.add(tf.layers.dropout({ rate: 0.2 }));
    m.add(tf.layers.dense({ units: 1, activation: 'sigmoid' }));

    const optimizer = tf.train.adam(typeof lr === 'number' ? lr : 1e-3);
    m.compile({ optimizer, loss: 'binaryCrossentropy', metrics: ['binaryAccuracy'] });

    this.model = m;
    this.log(`Built GRU: input [${timesteps}, ${features}], units=${U}, layers=${L}`);
    return this.model;
  }

  async fit({ xTrain, yTrain, epochs = 45, batchSize = 16, onEpoch, sampleWeight, validationSplit = 0.2, patience = 6 }) {
    if (!this.model) throw new Error('Build the model first.');
    const hist = { loss: [], val_loss: [] };
    let best = Number.POSITIVE_INFINITY, bad = 0;

    const callbacks = {
      onEpochEnd: async (epoch, logs) => {
        const L = logs.loss ?? NaN;
        const VL = (typeof logs.val_loss === 'number') ? logs.val_loss : NaN;
        hist.loss.push(L); if (!Number.isNaN(VL)) hist.val_loss.push(VL);
        this.log(`Epoch ${epoch+1}/${epochs} — loss=${L?.toFixed(4)} val_loss=${Number.isFinite(VL)?VL.toFixed(4):'—'}`);
        if (onEpoch) onEpoch(epoch, logs);

        if (Number.isFinite(VL)) {
          if (VL + 1e-6 < best) { best = VL; bad = 0; } else { bad += 1; }
          if (bad >= Math.max(2, patience|0)) {
            this.log(`Early stopping (no val_loss improvement ${patience} epochs).`);
            this.model.stopTraining = true;
          }
        }
        await tf.nextFrame();
      }
    };

    await this.model.fit(xTrain, yTrain, {
      epochs: Math.max(1, epochs|0),
      batchSize: Math.max(1, batchSize|0),
      shuffle: true,
      sampleWeight,
      validationSplit: Math.min(Math.max(validationSplit, 0.05), 0.4),
      callbacks
    });
    return hist; // return losses for plotting
  }

  predict(x) { if (!this.model) throw new Error('Model not built.'); return tf.tidy(()=>this.model.predict(x)); }

  async evaluate({ xTest, yTest, threshold = 0.5 }) {
    const yProb = this.predict(xTest);
    const yPred = tf.tidy(() => yProb.greaterEqual(threshold).toInt());
    const yTrue = yTest.toInt();

    const tp = tf.tidy(() => tf.logicalAnd(yPred.equal(1), yTrue.equal(1)).sum().arraySync());
    const tn = tf.tidy(() => tf.logicalAnd(yPred.equal(0), yTrue.equal(0)).sum().arraySync());
    const fp = tf.tidy(() => tf.logicalAnd(yPred.equal(1), yTrue.equal(0)).sum().arraySync());
    const fn = tf.tidy(() => tf.logicalAnd(yPred.equal(0), yTrue.equal(1)).sum().arraySync());

    const prec = tp / Math.max(1, tp + fp);
    const rec  = tp / Math.max(1, tp + fn);
    const f1   = 2 * prec * rec / Math.max(1e-9, (prec + rec));
    const auc  = await this.#rocAuc(yTest, yProb);

    yProb.dispose(); yPred.dispose(); yTrue.dispose();
    return { prec, rec, f1, auc, cm: { tp, tn, fp, fn } };
  }

  async save(name='tfjs-attrition-gru') { if (!this.model) throw new Error('Build the model first.'); await this.model.save(`indexeddb://${name}`); this.log(`Saved to indexeddb://${name}`); }
  async load(name='tfjs-attrition-gru') { this.dispose(); this.model = await tf.loadLayersModel(`indexeddb://${name}`); this.inputShape = this.model.inputs?.[0]?.shape?.slice(1); this.log(`Loaded model from indexeddb://${name}`); return this.model; }
  dispose(){ if (this.model){ this.model.dispose(); this.model=null; } }

  async #rocAuc(yTrue, yProb){
    const yt = (await yTrue.array()).map(r=>r[0]), yp = (await yPro
