'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { cn } from '@/lib/utils';

const LINKS = [
  { href: '/outliers/swing', match: '/outliers', label: 'Outliers' },
  { href: '/methodology', match: '/methodology', label: 'Methodology' },
];

export function NavLinks() {
  const pathname = usePathname();
  return (
    <nav aria-label="Main" className="ml-auto flex sm:ml-0">
      {LINKS.map((link) => {
        const active = pathname.startsWith(link.match);
        return (
          <Link
            key={link.href}
            href={link.href}
            aria-current={active ? 'page' : undefined}
            className={cn(
              'flex min-h-lg items-center border-b-2 px-xxs text-nav uppercase sm:px-xs',
              active ? 'border-primary text-ink' : 'border-transparent text-body hover:text-ink',
            )}
          >
            {link.label}
          </Link>
        );
      })}
    </nav>
  );
}
