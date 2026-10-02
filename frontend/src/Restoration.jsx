import { useState } from 'react'
import { post } from './api.js'
import { Bar, Card, Dropzone, ImagePanel, Segmented, Stat } from './ui.jsx'

const CONFIG = {
  universal: {
    title: 'Universal Restoration',
    endpoint: '/api/universal',
    blurb: 'One autoencoder restores clean, noisy, blurred, and occluded images without being told which one it received.',
  },
  hard: {
    title: 'Hard-Routed Restoration',
    endpoint: '/api/hard',
    blurb: 'A classifier names the corruption and one specialist autoencoder restores the image. Clean images skip restoration.',
  },
  soft: {
    title: 'Soft Mixture-of-Experts Restoration',
    endpoint: '/api/soft',
    blurb: 'A gate gives every branch a weight, and the output is the weighted sum of the identity branch and the three experts.',
  },
}

const CORRUPTIONS = [
  ['none', 'None (image is already corrupted or clean)'],
  ['salt_pepper', 'Salt-and-pepper noise'],
  ['blur', 'Gaussian blur'],
  ['occlusion', 'Rectangular occlusion'],
]
const SEVERITIES = [
  ['low', 'Low'],
  ['medium', 'Medium'],
  ['high', 'High'],
]
const NAMES = {
  clean: 'Clean',
  salt_pepper: 'Salt-and-pepper noise',
  blur: 'Gaussian blur',
  occlusion: 'Rectangular occlusion',
  identity: 'Identity (no restoration)',
}

function describe(s) {
  if (!s || s.type === 'none') return 'No corruption applied'
  if (s.type === 'salt_pepper') return `Salt-and-pepper, ${s.severity}, probability ${s.prob}`
  if (s.type === 'blur') return `Gaussian blur, ${s.severity}, kernel ${s.kernel_size}, sigma ${s.sigma}`
  return `Occlusion, ${s.severity}, ${s.rects.length} rectangle(s), ${(s.coverage * 100).toFixed(1)}% covered`
}

const db = (v) => (v == null ? 'n/a' : `${v} dB`)

export default function Restoration({ mode }) {
  const cfg = CONFIG[mode]
  const [file, setFile] = useState(null)
  const [corruption, setCorruption] = useState('salt_pepper')
  const [severity, setSeverity] = useState('medium')
  const [seed, setSeed] = useState(0)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState(null)
  const [result, setResult] = useState(null)

  async function run(useSeed = seed) {
    if (!file) {
      setError('Choose an image first.')
      return
    }
    setBusy(true)
    setError(null)
    const form = new FormData()
    form.append('image', file)
    form.append('corruption', corruption)
    form.append('severity', severity)
    form.append('seed', String(useSeed))
    try {
      setResult(await post(cfg.endpoint, form))
    } catch (e) {
      setError(e.message)
      setResult(null)
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="space-y-6">
      <Card title={`${cfg.title} settings`} subtitle={cfg.blurb}>
        <div className="grid gap-6 lg:grid-cols-3">
          <div>
            <div className="mb-2 text-xs font-semibold uppercase tracking-wide text-slate-500">Source image</div>
            <Dropzone file={file} onFile={(f) => { setFile(f); setResult(null) }} />
          </div>
          <div className="space-y-4">
            <div>
              <div className="mb-2 text-xs font-semibold uppercase tracking-wide text-slate-500">Corruption to apply</div>
              <select
                value={corruption}
                onChange={(e) => setCorruption(e.target.value)}
                className="w-full rounded-xl border border-indigo-200 bg-white px-3 py-2 text-sm"
              >
                {CORRUPTIONS.map(([key, label]) => (
                  <option key={key} value={key}>{label}</option>
                ))}
              </select>
            </div>
            <div>
              <div className="mb-2 text-xs font-semibold uppercase tracking-wide text-slate-500">Severity</div>
              <Segmented options={SEVERITIES} value={severity} onChange={setSeverity} />
            </div>
          </div>
          <div className="flex flex-col justify-end gap-3">
            <button
              type="button"
              disabled={busy}
              onClick={() => run()}
              className="rounded-xl bg-indigo-600 px-4 py-3 font-semibold text-white shadow hover:bg-indigo-700 disabled:opacity-60"
            >
              {busy ? 'Working...' : 'Restore image'}
            </button>
            {corruption !== 'none' && (
              <button
                type="button"
                disabled={busy}
                onClick={() => {
                  const next = seed + 1
                  setSeed(next)
                  run(next)
                }}
                className="rounded-xl bg-white px-4 py-2 text-sm font-medium text-indigo-700 ring-1 ring-indigo-200 hover:bg-indigo-50 disabled:opacity-60"
              >
                Try a different random pattern
              </button>
            )}
          </div>
        </div>
        {error && <p className="mt-4 rounded-xl bg-red-50 px-3 py-2 text-sm text-red-700">{error}</p>}
      </Card>

      {result && (
        <>
          <div className="grid gap-4 md:grid-cols-3">
            <ImagePanel title="Original (128 x 128)" src={result.input} />
            <ImagePanel title="Corrupted input" src={result.corrupted} />
            <ImagePanel title="Restored output" src={result.restored} download="restored.png" />
          </div>

          <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
            <Stat label="Inference time" value={`${result.inference_ms} ms`} hint="ONNX Runtime on CPU" />
            <Stat label="PSNR of input" value={db(result.psnr_input_vs_original)} hint="corrupted vs original" />
            <Stat label="PSNR of output" value={db(result.psnr_restored_vs_original)} hint="restored vs original" />
            <Stat label="Corruption settings" value={describe(result.settings)} compact />
          </div>

          {mode === 'hard' && (
            <Card title="Router" subtitle="Classifier probabilities and the expert that was selected.">
              <div className="grid gap-6 lg:grid-cols-2">
                <div className="space-y-2">
                  {Object.entries(result.probabilities).map(([key, value]) => (
                    <Bar key={key} label={NAMES[key]} value={value} highlight={key === result.predicted} />
                  ))}
                </div>
                <div className="grid gap-4 sm:grid-cols-2">
                  <Stat label="Predicted corruption" value={NAMES[result.predicted]} compact />
                  <Stat label="Selected expert" value={result.expert} compact />
                  <Stat label="Classifier time" value={`${result.classifier_ms} ms`} />
                  <Stat label="Expert time" value={`${result.expert_ms} ms`} />
                </div>
              </div>
            </Card>
          )}

          {mode === 'soft' && (
            <Card title="Routing weights" subtitle="The weights always sum to 1. The strongest branch is highlighted.">
              <div className="grid gap-6 lg:grid-cols-2">
                <div className="space-y-2">
                  {Object.entries(result.weights).map(([key, value]) => (
                    <Bar key={key} label={NAMES[key]} value={value} highlight={key === result.dominant} />
                  ))}
                </div>
                <Stat label="Dominant branch" value={NAMES[result.dominant]} compact />
              </div>
            </Card>
          )}
        </>
      )}
    </div>
  )
}
