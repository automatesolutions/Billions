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
              'group relative flex min-h-lg items-center px-xxs text-nav uppercase transition-colors duration-300 sm:px-xs',
              active ? 'text-ink' : 'text-body hover:text-ink',
            )}
          >
            {link.label}
            {/* Racing underline: grows from the left on hover, stays full on the current page. */}
            <span
              aria-hidden
              className={cn(
                'absolute inset-x-xxs bottom-0 h-[2px] origin-left bg-primary transition-transform duration-500 ease-out sm:inset-x-xs',
                active ? 'scale-x-100' : 'scale-x-0 group-hover:scale-x-100',
              )}
            />
          </Link>
        );
      })}
    </nav>
  );
}
