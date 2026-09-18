import { useEffect, useState } from 'react'
import { fetchCandleHistory, fetchStatus } from '../api/client'
import type {
  CandlePoint,
  ConnectionState,
  SentimentReading,
  StatusResponse,
  WireEvent,
} from '../types/events'

function toCandlePoint(event: WireEvent): CandlePoint | null {
  if (
    event.candle_timestamp === null ||
    event.open === null ||
    event.high === null ||
    event.low === null ||
    event.close === null ||
    event.volume === null
  ) {
    return null
  }
  return {
    time: Math.floor(new Date(event.candle_timestamp).getTime() / 1000),
    open: event.open,
    high: event.high,
    low: event.low,
    close: event.close,
    volume: event.volume,
  }
}

// A new candle event either extends the series (new bar) or, since the
// backend's most recent bar can still be forming, updates the last one in
// place (same timestamp) — see LiveCandleFeed's docstring on that.
function appendCandle(candles: CandlePoint[], next: CandlePoint): CandlePoint[] {
  const last = candles[candles.length - 1]
  if (last && last.time === next.time) {
    return [...candles.slice(0, -1), next]
  }
  return [...candles, next]
}

export interface SentimentSocketState {
  connection: ConnectionState
  status: StatusResponse | null
  // Keyed by event.source ("momentum_v0", "flag_v0", ...) — more than one
  // model can be running at once, each publishing its own SentimentEvent
  // stream, so a single latest-sentiment value would have the later model
  // silently clobber the earlier one's reading.
  sentiments: Record<string, SentimentReading>
  candles: CandlePoint[]
  ticker: string | null
  isReplay: boolean
}

/**
 * Owns the single WebSocket connection to /ws (reconnecting every 2s on
 * drop, same as the original static page) plus the one-time REST seed from
 * /api/status and /api/candles — a page that only listened to the socket
 * would start with an empty chart and take a while to fill in, since a
 * candle event only arrives once per candle close.
 */
export function useSentimentSocket(): SentimentSocketState {
  const [connection, setConnection] = useState<ConnectionState>('connecting')
  const [status, setStatus] = useState<StatusResponse | null>(null)
  const [sentiments, setSentiments] = useState<Record<string, SentimentReading>>({})
  const [candles, setCandles] = useState<CandlePoint[]>([])
  const [ticker, setTicker] = useState<string | null>(null)
  const [isReplay, setIsReplay] = useState(true)

  useEffect(() => {
    fetchStatus()
      .then(setStatus)
      .catch(() => {})
    fetchCandleHistory()
      .then((history) => {
        const seeded = history
          .map(toCandlePoint)
          .filter((c): c is CandlePoint => c !== null)
          .sort((a, b) => a.time - b.time)
        setCandles(seeded)
      })
      .catch(() => {})
  }, [])

  useEffect(() => {
    let cancelled = false
    let ws: WebSocket | null = null
    let retryTimer: number | undefined

    function connect() {
      if (cancelled) return
      const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:'
      ws = new WebSocket(`${protocol}//${location.host}/ws`)

      ws.onopen = () => setConnection('connected')
      ws.onclose = () => {
        setConnection('disconnected')
        retryTimer = window.setTimeout(connect, 2000)
      }
      ws.onerror = () => ws?.close()

      ws.onmessage = (msg) => {
        const event: WireEvent = JSON.parse(msg.data)
        if (event.ticker) setTicker(event.ticker)
        setIsReplay(event.is_replay)

        if (event.event_type === 'sentiment' && event.sentiment !== null && event.confidence !== null) {
          const reading: SentimentReading = {
            ticker: event.ticker ?? '',
            source: event.source,
            sentiment: event.sentiment,
            confidence: event.confidence,
            isReplay: event.is_replay,
            asOf: event.as_of,
          }
          setSentiments((prev) => ({ ...prev, [event.source]: reading }))
        } else if (event.event_type === 'candle') {
          const point = toCandlePoint(event)
          if (point) setCandles((prev) => appendCandle(prev, point))
        }
      }
    }

    connect()
    return () => {
      cancelled = true
      window.clearTimeout(retryTimer)
      ws?.close()
    }
  }, [])

  return { connection, status, sentiments, candles, ticker, isReplay }
}
