import type { CandlePoint } from '../types/events'
import { WidgetCard } from './WidgetCard'

export function PriceWidget({ candles }: { candles: CandlePoint[] }) {
  const latest = candles[candles.length - 1]
  const previous = candles[candles.length - 2]

  if (!latest) {
    return (
      <WidgetCard title="Price">
        <p className="widget-empty">Waiting for the first candle…</p>
      </WidgetCard>
    )
  }

  const delta = previous ? latest.close - previous.close : null
  const deltaPct = previous ? (delta! / previous.close) * 100 : null
  const deltaColor = delta === null ? 'var(--color-neutral)' : delta > 0 ? 'var(--color-bullish)' : delta < 0 ? 'var(--color-bearish)' : 'var(--color-neutral)'

  return (
    <WidgetCard title="Price">
      <div className="price-widget__value">{latest.close.toFixed(2)}</div>
      {delta !== null && (
        <div className="price-widget__delta" style={{ color: deltaColor }}>
          {delta >= 0 ? '+' : ''}
          {delta.toFixed(2)} ({deltaPct! >= 0 ? '+' : ''}
          {deltaPct!.toFixed(2)}%) vs previous candle
        </div>
      )}
      <div className="price-widget__meta">
        last candle: {new Date(latest.time * 1000).toLocaleTimeString()}
      </div>
    </WidgetCard>
  )
}
