import type { ReactNode } from 'react'

export function WidgetCard({ title, children }: { title: string; children: ReactNode }) {
  return (
    <div className="widget-card">
      <h2 className="widget-card__title">{title}</h2>
      {children}
    </div>
  )
}
