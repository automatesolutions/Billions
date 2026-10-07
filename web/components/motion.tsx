'use client';

import { useEffect, useRef, type RefObject } from 'react';

/** GSAP is loaded on demand so it never blocks first paint. */
const loadGsap = () => import('gsap').then((m) => m.gsap);

function prefersReducedMotion() {
  return typeof window !== 'undefined' && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
}

/**
 * Fades and lifts every `[data-reveal]` element inside `ref` once, on first render.
 * Elements are visible by default; the animation only runs when motion is allowed.
 */
export function useReveal(ref: RefObject<HTMLElement | null>, ready = true) {
  const done = useRef(false);
  useEffect(() => {
    const root = ref.current;
    if (!ready || done.current || !root || prefersReducedMotion()) return;
    done.current = true;
    let cancelled = false;
    loadGsap().then((gsap) => {
      if (cancelled) return;
      // Only animate what is rendered and inside the first screen; the rest is already in place.
      const targets = Array.from(root.querySelectorAll<HTMLElement>('[data-reveal]')).filter((el) => {
        if (el.offsetParent === null) return false;
        return el.getBoundingClientRect().top < window.innerHeight;
      });
      if (!targets.length) return;
      // Total stagger is capped, so long lists finish within ~0.7s.
      gsap.from(targets, { y: 8, opacity: 0, duration: 0.4, ease: 'power2.out', stagger: { amount: 0.3 }, clearProps: 'all' });
    });
    return () => {
      cancelled = true;
    };
  }, [ref, ready]);
}

/**
 * Shows `value` formatted. On first mount it counts up from zero (0.6s).
 * The server-rendered text is already the final value.
 */
export function CountUp({ value, format, className }: { value: number; format: (n: number) => string; className?: string }) {
  const ref = useRef<HTMLSpanElement>(null);
  const animated = useRef(false);

  useEffect(() => {
    const el = ref.current;
    if (!el || animated.current || !Number.isFinite(value) || prefersReducedMotion()) return;
    animated.current = true;
    let cancelled = false;
    loadGsap().then((gsap) => {
      if (cancelled) return;
      const state = { n: 0 };
      gsap.to(state, {
        n: value,
        duration: 0.6,
        ease: 'power2.out',
        onUpdate: () => {
          el.textContent = format(state.n);
        },
        onComplete: () => {
          el.textContent = format(value);
        },
      });
    });
    return () => {
      cancelled = true;
    };
  }, [value, format]);

  return (
    <span ref={ref} className={className}>
      {format(value)}
    </span>
  );
}
