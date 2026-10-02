export async function post(path, form) {
  const res = await fetch(path, { method: 'POST', body: form })
  const data = await res.json().catch(() => ({}))
  if (!res.ok) {
    const detail = typeof data.detail === 'string' ? data.detail : JSON.stringify(data.detail)
    throw new Error(detail || `Request failed (${res.status})`)
  }
  return data
}
