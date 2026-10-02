import { useState } from 'react'

export function Card({ title, subtitle, children, className = '' }) {
  return (
    <section className={`rounded-2xl bg-white p-5 shadow-sm ring-1 ring-indigo-100 ${className}`}>
      {title && <h2 className="text-lg font-semibold text-slate-900">{title}</h2>}
      {subtitle && <p className="mt-1 text-sm text-slate-500">{subtitle}</p>}
      <div className={title ? 'mt-4' : ''}>{children}</div>
    </section>
  )
}

export function Segmented({ options, value, onChange }) {
  return (
    <div className="inline-flex rounded-xl bg-indigo-50 p-1">
      {options.map(([key, label]) => (
        <button
          key={key}
          type="button"
          onClick={() => onChange(key)}
          className={`rounded-lg px-4 py-1.5 text-sm font-medium transition ${
            value === key ? 'bg-indigo-600 text-white shadow' : 'text-slate-600 hover:text-indigo-700'
          }`}
        >
          {label}
        </button>
      ))}
    </div>
  )
}

export function Bar({ label, value, highlight }) {
  const pct = Math.round(value * 1000) / 10
  return (
    <div className={`rounded-xl px-3 py-2 ${highlight ? 'bg-indigo-50 ring-1 ring-indigo-200' : ''}`}>
      <div className="flex justify-between text-sm">
        <span className={highlight ? 'font-semibold text-indigo-700' : 'text-slate-700'}>{label}</span>
        <span className="tabular-nums text-slate-600">{pct.toFixed(1)}%</span>
      </div>
      <div className="mt-1 h-2 rounded-full bg-slate-100">
        <div
          className={`h-2 rounded-full ${highlight ? 'bg-indigo-600' : 'bg-slate-400'}`}
          style={{ width: `${pct}%` }}
        />
      </div>
    </div>
  )
}

export function Stat({ label, value, hint, compact }) {
  return (
    <div className="rounded-2xl bg-white p-4 shadow-sm ring-1 ring-indigo-100">
      <div className="text-xs font-semibold uppercase tracking-wide text-slate-500">{label}</div>
      <div className={`mt-2 font-semibold text-slate-900 ${compact ? 'text-sm' : 'text-2xl'}`}>{value}</div>
      {hint && <div className="mt-1 text-xs text-slate-500">{hint}</div>}
    </div>
  )
}

export function ImagePanel({ title, src, download }) {
  return (
    <figure className="overflow-hidden rounded-2xl bg-white ring-1 ring-indigo-100">
      <figcaption className="flex items-center justify-between px-4 py-2 text-sm font-medium text-slate-700">
        <span>{title}</span>
        {src && download && (
          <a href={src} download={download} className="text-indigo-600 hover:underline">
            Download
          </a>
        )}
      </figcaption>
      <div className="aspect-square bg-slate-100">
        {src ? (
          <img src={src} alt={title} className="h-full w-full object-contain" />
        ) : (
          <div className="flex h-full items-center justify-center text-sm text-slate-400">No image yet</div>
        )}
      </div>
    </figure>
  )
}

export function Dropzone({ file, onFile }) {
  const [over, setOver] = useState(false)
  return (
    <label
      onDragOver={(e) => {
        e.preventDefault()
        setOver(true)
      }}
      onDragLeave={() => setOver(false)}
      onDrop={(e) => {
        e.preventDefault()
        setOver(false)
        const f = e.dataTransfer.files?.[0]
        if (f) onFile(f)
      }}
      className={`flex cursor-pointer flex-col items-center justify-center rounded-2xl border-2 border-dashed p-6 text-center transition ${
        over ? 'border-indigo-500 bg-indigo-50' : 'border-indigo-200 bg-white hover:bg-indigo-50'
      }`}
    >
      <input
        type="file"
        accept="image/jpeg,image/png,image/webp"
        className="hidden"
        onChange={(e) => {
          const f = e.target.files?.[0]
          if (f) onFile(f)
        }}
      />
      <span className="text-sm font-medium text-indigo-700">
        {file ? file.name : 'Drag an image here or click to choose'}
      </span>
      <span className="mt-1 text-xs text-slate-500">JPEG, PNG, or WebP, up to 10 MB</span>
    </label>
  )
}
