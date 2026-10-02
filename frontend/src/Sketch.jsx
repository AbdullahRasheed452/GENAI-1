import { useEffect, useMemo, useRef, useState } from 'react'
import { post } from './api.js'
import { Card, Dropzone, ImagePanel, Segmented, Stat } from './ui.jsx'

const STYLES = [
  [1, 'Style 1'],
  [2, 'Style 2'],
  [3, 'Style 3'],
]

export default function Sketch() {
  const [file, setFile] = useState(null)
  const [style, setStyle] = useState(1)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState(null)
  const [result, setResult] = useState(null)
  const [camera, setCamera] = useState(false)
  const videoRef = useRef(null)
  const streamRef = useRef(null)
  const preview = useMemo(() => (file ? URL.createObjectURL(file) : null), [file])

  useEffect(() => {
    if (camera && videoRef.current) videoRef.current.srcObject = streamRef.current
  }, [camera])
  useEffect(() => () => streamRef.current?.getTracks().forEach((t) => t.stop()), [])

  async function startCamera() {
    setError(null)
    try {
      streamRef.current = await navigator.mediaDevices.getUserMedia({ video: true })
      setCamera(true)
    } catch (e) {
      setError(`Could not open the webcam: ${e.message}`)
    }
  }

  function stopCamera() {
    streamRef.current?.getTracks().forEach((t) => t.stop())
    streamRef.current = null
    setCamera(false)
  }

  function capture() {
    const video = videoRef.current
    const canvas = document.createElement('canvas')
    canvas.width = video.videoWidth
    canvas.height = video.videoHeight
    canvas.getContext('2d').drawImage(video, 0, 0)
    canvas.toBlob((blob) => {
      setFile(new File([blob], 'webcam.png', { type: 'image/png' }))
      setResult(null)
      stopCamera()
    }, 'image/png')
  }

  async function generate() {
    if (!file) {
      setError('Upload a photo or capture one with the webcam first.')
      return
    }
    setBusy(true)
    setError(null)
    const form = new FormData()
    form.append('image', file)
    form.append('style', String(style))
    try {
      setResult(await post('/api/sketch', form))
    } catch (e) {
      setError(e.message)
      setResult(null)
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="space-y-6">
      <Card
        title="Face-to-Sketch Generator"
        subtitle="A conditional GAN turns a face photo into a pencil sketch in one of three styles."
      >
        <div className="grid gap-6 lg:grid-cols-3">
          <div className="space-y-3">
            <div className="text-xs font-semibold uppercase tracking-wide text-slate-500">Photo</div>
            {camera ? (
              <div className="space-y-2">
                <video ref={videoRef} autoPlay playsInline muted className="w-full rounded-2xl bg-black" />
                <div className="flex gap-2">
                  <button type="button" onClick={capture} className="flex-1 rounded-xl bg-indigo-600 px-3 py-2 text-sm font-semibold text-white">
                    Capture photo
                  </button>
                  <button type="button" onClick={stopCamera} className="rounded-xl bg-white px-3 py-2 text-sm ring-1 ring-indigo-200">
                    Cancel
                  </button>
                </div>
              </div>
            ) : (
              <>
                <Dropzone file={file} onFile={(f) => { setFile(f); setResult(null) }} />
                <button type="button" onClick={startCamera} className="w-full rounded-xl bg-white px-3 py-2 text-sm font-medium text-indigo-700 ring-1 ring-indigo-200 hover:bg-indigo-50">
                  Use webcam
                </button>
              </>
            )}
          </div>
          <div>
            <div className="mb-2 text-xs font-semibold uppercase tracking-wide text-slate-500">Sketch style</div>
            <Segmented options={STYLES} value={style} onChange={setStyle} />
          </div>
          <div className="flex flex-col justify-end">
            <button
              type="button"
              disabled={busy}
              onClick={generate}
              className="rounded-xl bg-indigo-600 px-4 py-3 font-semibold text-white shadow hover:bg-indigo-700 disabled:opacity-60"
            >
              {busy ? 'Working...' : 'Generate sketch'}
            </button>
          </div>
        </div>
        {error && <p className="mt-4 rounded-xl bg-red-50 px-3 py-2 text-sm text-red-700">{error}</p>}
      </Card>

      <div className="grid gap-4 md:grid-cols-2">
        <ImagePanel title="Original photo" src={result?.photo || preview} />
        <ImagePanel title="Generated sketch" src={result?.sketch} download="sketch.png" />
      </div>

      {result && (
        <div className="grid gap-4 sm:grid-cols-2">
          <Stat label="Inference time" value={`${result.inference_ms} ms`} hint="ONNX Runtime on CPU" />
          <Stat label="Style used" value={`Style ${result.style}`} />
        </div>
      )}
    </div>
  )
}
