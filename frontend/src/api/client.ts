import type { StatusResponse, WireEvent } from '../types/events'

export async function fetchStatus(): Promise<StatusResponse> {
  const res = await fetch('/api/status')
  if (!res.ok) throw new Error(`GET /api/status failed: ${res.status}`)
  return res.json()
}

export async function fetchCandleHistory(): Promise<WireEvent[]> {
  const res = await fetch('/api/candles')
  if (!res.ok) throw new Error(`GET /api/candles failed: ${res.status}`)
  return res.json()
}
