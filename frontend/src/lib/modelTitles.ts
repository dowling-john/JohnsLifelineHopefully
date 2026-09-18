// Display title for each backend model source. A model not listed here
// (a new indicator added on the backend before this map is updated)
// still renders — just under its raw source name — rather than being
// silently dropped from the grid.
export const MODEL_TITLES: Record<string, string> = {
  momentum_v0: 'Sentiment (Momentum)',
  flag_v0: 'Sentiment (Flag)',
  rsi_v0: 'Mean Reversion (RSI)',
  squeeze_v0: 'Volatility Squeeze',
  divergence_v0: 'Volume Divergence',
  support_resistance_v0: 'Support / Resistance',
  candlestick_v0: 'Candlestick Reversal',
}

export function titleFor(modelName: string): string {
  return MODEL_TITLES[modelName] ?? modelName
}
