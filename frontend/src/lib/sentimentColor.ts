// Ported from the original static page's colourFor().
export function sentimentColor(sentiment: number): string {
  if (sentiment > 0.15) return 'var(--color-bullish)'
  if (sentiment < -0.15) return 'var(--color-bearish)'
  return 'var(--color-neutral)'
}
