import type { ConnectionState } from '../types/events'

const CONNECTION_LABEL: Record<ConnectionState, string> = {
  connecting: 'connecting…',
  connected: 'connected',
  disconnected: 'disconnected — retrying',
}

export function TickerHeader({
  ticker,
  isReplay,
  connection,
}: {
  ticker: string | null
  isReplay: boolean
  connection: ConnectionState
}) {
  return (
    <header className="header">
      <span className="header__ticker">{ticker ?? '—'}</span>
      <span className={`mode-badge ${isReplay ? 'mode-badge--replay' : 'mode-badge--live'}`}>
        {isReplay ? 'REPLAY' : 'LIVE'}
      </span>
      <span className={`connection-status connection-status--${connection}`}>
        <span className="connection-status__dot" />
        {CONNECTION_LABEL[connection]}
      </span>
    </header>
  )
}
