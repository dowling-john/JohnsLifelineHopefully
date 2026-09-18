// Mirrors decision_engine/ui_server.py's _serialize_event() output — the
// same shape goes out over the WebSocket and comes back from GET
// /api/candles, so one type covers both.
export interface WireEvent {
  event_type: 'candle' | 'sentiment'
  source: string
  ticker: string | null
  sentiment: number | null
  confidence: number | null
  open: number | null
  high: number | null
  low: number | null
  close: number | null
  volume: number | null
  candle_timestamp: string | null
  as_of: string
  is_replay: boolean
}

export interface ModelStatus {
  name: string
  window_size: number
}

export interface StatusResponse {
  ticker: string | null
  interval: string | null
  mode: 'live' | 'replay'
  poll_seconds: number | null
  models: ModelStatus[]
}

export interface CandlePoint {
  time: number // unix seconds, what lightweight-charts expects
  open: number
  high: number
  low: number
  close: number
  volume: number
}

export interface SentimentReading {
  ticker: string
  source: string
  sentiment: number
  confidence: number
  isReplay: boolean
  asOf: string
}

export type ConnectionState = 'connecting' | 'connected' | 'disconnected'
