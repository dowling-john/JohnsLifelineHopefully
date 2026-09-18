import './App.css'
import { useSentimentSocket } from './hooks/useSentimentSocket'
import { TickerHeader } from './components/TickerHeader'
import { SentimentWidget } from './components/SentimentWidget'
import { PriceWidget } from './components/PriceWidget'
import { CandleChartWidget } from './components/CandleChartWidget'
import { titleFor } from './lib/modelTitles'

function App() {
  const { connection, status, sentiments, candles, ticker, isReplay } = useSentimentSocket()

  return (
    <div className="app">
      <TickerHeader ticker={ticker ?? status?.ticker ?? null} isReplay={isReplay} connection={connection} />
      {/* Data-driven from /api/status's models list rather than one
          hardcoded widget per indicator — a future indicator just needs a
          backend model + an entry in modelTitles.ts, not a change here. */}
      <div className="indicator-grid">
        {(status?.models ?? []).map((m) => (
          <SentimentWidget
            key={m.name}
            title={titleFor(m.name)}
            sentiment={sentiments[m.name] ?? null}
            windowSize={m.window_size}
          />
        ))}
      </div>
      <div className="widget-grid">
        <PriceWidget candles={candles} />
        <CandleChartWidget candles={candles} />
      </div>
    </div>
  )
}

export default App
