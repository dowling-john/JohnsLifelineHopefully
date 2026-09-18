import type { SentimentReading } from '../types/events'
import { sentimentColor } from '../lib/sentimentColor'
import { WidgetCard } from './WidgetCard'

export function SentimentWidget({
  title,
  sentiment,
  windowSize,
}: {
  title: string
  sentiment: SentimentReading | null
  windowSize: number | null
}) {
  return (
    <WidgetCard title={title}>
      {sentiment === null ? (
        <p className="widget-empty">
          {windowSize ? `Waiting for ${windowSize} candles to fill the window…` : 'Waiting for the first prediction…'}
        </p>
      ) : (
        <div className="sentiment-widget">
          <div className="sentiment-widget__value" style={{ color: sentimentColor(sentiment.sentiment) }}>
            {sentiment.sentiment.toFixed(2)}
          </div>
          <div className="sentiment-widget__confidence">
            confidence: {(sentiment.confidence * 100).toFixed(0)}%
          </div>
          <div className="sentiment-widget__bar-track">
            <div
              className="sentiment-widget__bar-fill"
              style={{
                width: `${((sentiment.sentiment + 1) / 2) * 100}%`,
                background: sentimentColor(sentiment.sentiment),
              }}
            />
          </div>
        </div>
      )}
    </WidgetCard>
  )
}
