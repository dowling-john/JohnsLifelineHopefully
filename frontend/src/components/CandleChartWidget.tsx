import { useEffect, useRef } from 'react'
import { CandlestickSeries, ColorType, createChart, type IChartApi, type ISeriesApi } from 'lightweight-charts'
import type { CandlePoint } from '../types/events'
import { WidgetCard } from './WidgetCard'

export function CandleChartWidget({ candles }: { candles: CandlePoint[] }) {
  const containerRef = useRef<HTMLDivElement>(null)
  const chartRef = useRef<IChartApi | null>(null)
  const seriesRef = useRef<ISeriesApi<'Candlestick'> | null>(null)

  // Create the chart once; it's resized/populated by the effects below.
  useEffect(() => {
    const container = containerRef.current
    if (!container) return

    const chart = createChart(container, {
      layout: {
        background: { type: ColorType.Solid, color: 'transparent' },
        textColor: '#9aa4b2',
      },
      grid: {
        vertLines: { color: '#1c1f26' },
        horzLines: { color: '#1c1f26' },
      },
      rightPriceScale: { borderColor: '#232733' },
      timeScale: { borderColor: '#232733', timeVisible: true, secondsVisible: false },
      autoSize: true,
    })
    const series = chart.addSeries(CandlestickSeries, {
      upColor: '#3ddc84',
      downColor: '#ff5c5c',
      borderVisible: false,
      wickUpColor: '#3ddc84',
      wickDownColor: '#ff5c5c',
    })

    chartRef.current = chart
    seriesRef.current = series

    return () => {
      chart.remove()
      chartRef.current = null
      seriesRef.current = null
    }
  }, [])

  useEffect(() => {
    if (!seriesRef.current || candles.length === 0) return
    seriesRef.current.setData(
      candles.map((c) => ({ time: c.time as never, open: c.open, high: c.high, low: c.low, close: c.close })),
    )
    chartRef.current?.timeScale().fitContent()
  }, [candles])

  return (
    <WidgetCard title="Candles">
      {/* Always mounted (never conditionally rendered) — the chart is
          created imperatively into this node exactly once on mount, so if
          this were swapped out for a placeholder while candles is still
          empty, containerRef would be null when that effect runs and the
          chart would never get created once data arrives. */}
      <div className="chart-widget__wrap">
        {candles.length === 0 && <p className="widget-empty chart-widget__empty-overlay">Waiting for candle history…</p>}
        <div className="chart-widget__canvas" ref={containerRef} />
      </div>
    </WidgetCard>
  )
}
