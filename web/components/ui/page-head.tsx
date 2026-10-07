import type { ReactNode } from 'react';

/**
 * Top band for inner pages: a smaller cousin of the home hero.
 * Full-bleed, faint timing-screen grid, a Rosso Corsa glow in the corner, display type on the left.
 */
export function PageHead({ eyebrow, title, children }: { eyebrow: ReactNode; title: ReactNode; children?: ReactNode }) {
  return (
    <header className="full-bleed relative isolate -mt-md overflow-hidden border-b border-hairline">
      <div aria-hidden className="absolute inset-0 -z-10">
        <div className="bg-grid absolute inset-0 [mask-image:linear-gradient(to_left,black,transparent_70%)]" />
        <div className="bg-livery absolute -right-xl -top-xxl size-[420px] rounded-full opacity-25 blur-3xl" />
      </div>
      <div className="mx-auto flex max-w-content flex-col gap-xs px-xs pb-lg pt-xl sm:px-md">
        <p className="flex items-center gap-xs text-caption-upper uppercase text-body">
          <span className="stripe" />
          {eyebrow}
        </p>
        <h1 className="text-display-lg sm:text-display-xl">{title}</h1>
        {children}
      </div>
    </header>
  );
}
