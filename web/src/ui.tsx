import type { ReactNode } from 'react'
import { ArrowUpRight } from 'lucide-react'

export function Badge({ children, tone = 'neutral' }: { children: ReactNode; tone?: string }) {
  return <span className={'badge tone-' + tone}>{children}</span>
}

export function PageHeader({ title, description, tag, children }: {
  title: string; description: string; tag: string; children?: ReactNode
}) {
  return <header className="page-head">
    <div><span className="eyebrow">{tag}</span><h1>{title}</h1><p>{description}</p></div>{children}
  </header>
}

export function Panel({ title, description, accessory, children, className = '' }: {
  title: string; description?: string; accessory?: ReactNode; children: ReactNode; className?: string
}) {
  return <section className={'panel ' + className}>
    <div className="panel-head"><div><h2>{title}</h2>{description && <p>{description}</p>}</div>{accessory}</div>
    {children}
  </section>
}

export function External({ href, children }: { href: string; children: ReactNode }) {
  return <a className="text-link" href={href} target="_blank" rel="noopener noreferrer">{children}<ArrowUpRight size={15} aria-hidden="true" /></a>
}
