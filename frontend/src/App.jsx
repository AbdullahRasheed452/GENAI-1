import { useEffect, useState } from 'react'
import Restoration from './Restoration.jsx'
import Sketch from './Sketch.jsx'

const PAGES = [
  ['universal', 'Universal Restoration'],
  ['hard', 'Hard-Routed Restoration'],
  ['soft', 'Soft Mixture-of-Experts'],
  ['sketch', 'Face-to-Sketch Generator'],
]

export default function App() {
  const [page, setPage] = useState('universal')
  const [health, setHealth] = useState(null)
  const [offline, setOffline] = useState(false)

  useEffect(() => {
    fetch('/api/health')
      .then((r) => r.json())
      .then(setHealth)
      .catch(() => setOffline(true))
  }, [])

  const title = PAGES.find(([key]) => key === page)[1]
  const missing = health?.models_missing?.length > 0

  return (
    <div className="flex min-h-screen flex-col bg-indigo-50 text-slate-800 md:flex-row">
      <aside className="border-b border-indigo-100 bg-white p-4 md:w-64 md:shrink-0 md:border-b-0 md:border-r">
        <div className="mb-6">
          <div className="text-lg font-bold text-slate-900">Restoration and Sketch Studio</div>
          <div className="text-xs text-slate-500">Generative AI, Assignment 1</div>
        </div>
        <nav className="flex gap-2 overflow-x-auto md:flex-col">
          {PAGES.map(([key, label]) => (
            <button
              key={key}
              type="button"
              onClick={() => setPage(key)}
              className={`whitespace-nowrap rounded-xl px-4 py-2.5 text-left text-sm font-medium transition ${
                page === key ? 'bg-indigo-600 text-white shadow' : 'text-slate-700 hover:bg-indigo-50'
              }`}
            >
              {label}
            </button>
          ))}
        </nav>
        <div className="mt-6 hidden rounded-xl bg-indigo-50 p-3 text-xs text-slate-600 md:block">
          Runtime: ONNX Runtime on CPU. All models are exported from PyTorch and checked against it.
        </div>
      </aside>

      <main className="flex-1 p-4 md:p-8">
        <header className="mb-6 flex flex-wrap items-center justify-between gap-3">
          <h1 className="text-2xl font-semibold text-slate-900">{title}</h1>
          {offline && <span className="rounded-full bg-red-100 px-3 py-1 text-sm text-red-800">Backend offline</span>}
          {health && !missing && (
            <span className="rounded-full bg-green-100 px-3 py-1 text-sm text-green-800">
              Backend connected, {health.models_loaded.length} models loaded
            </span>
          )}
          {health && missing && (
            <span className="rounded-full bg-amber-100 px-3 py-1 text-sm text-amber-800">
              Missing models: {health.models_missing.join(', ')}
            </span>
          )}
        </header>
        {page === 'sketch' ? <Sketch /> : <Restoration key={page} mode={page} />}
      </main>
    </div>
  )
}
