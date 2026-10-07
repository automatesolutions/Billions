import type { Metadata } from 'next';
import type { ReactNode } from 'react';

export const metadata: Metadata = {
  title: 'Methodology',
  description: 'How BILLIONS finds outliers and tests each stock, in plain language, with the limits of the data.',
  alternates: { canonical: '/methodology' },
};

const SECTIONS = [
  { id: 'outliers', title: 'How outliers are found' },
  { id: 'analysis', title: 'How a stock is analyzed' },
  { id: 'validation', title: 'How the models are checked' },
  { id: 'limits', title: 'Limits of the data' },
];

function Section({ id, title, children }: { id: string; title: string; children: ReactNode }) {
  return (
    <section id={id} aria-labelledby={`${id}-title`} className="flex flex-col gap-xs border-t border-hairline pt-md">
      <h2 id={`${id}-title`} className="text-display-md">
        {title}
      </h2>
      <div className="flex max-w-prose flex-col gap-xs text-body-md [&_strong]:font-semibold [&_strong]:text-ink">{children}</div>
    </section>
  );
}

function Term({ name, children }: { name: string; children: ReactNode }) {
  return (
    <div className="border-l-2 border-hairline pl-xs">
      <dt className="text-title-sm text-ink">{name}</dt>
      <dd>{children}</dd>
    </div>
  );
}

export default function MethodologyPage() {
  return (
    <article className="flex flex-col gap-lg">
      <header className="flex flex-col gap-xs">
        <p className="text-caption-upper uppercase text-body">Methodology</p>
        <h1 className="text-display-lg sm:text-display-xl">What the numbers mean</h1>
        <p className="max-w-prose text-title-sm text-body">
          BILLIONS measures how unusual a stock&apos;s recent move is, then tests whether its past returns show any edge.
          It describes data. It doesn&apos;t tell you what to do.
        </p>
        <nav aria-label="On this page" className="flex flex-wrap gap-x-sm gap-y-xxs pt-xs text-nav uppercase">
          {SECTIONS.map((s) => (
            <a key={s.id} href={`#${s.id}`} className="flex min-h-lg items-center text-body hover:text-ink">
              {s.title}
            </a>
          ))}
        </nav>
      </header>

      <Section id="outliers" title="How outliers are found">
        <p>
          The scan starts from every common stock listed on NASDAQ, about 4,000 names. It keeps stocks that trade at least a
          set dollar amount a day (the median of the last 20 sessions) and cost at least $3. Then it keeps the 1,000 most
          traded.
        </p>
        <p>Each strategy compares two windows, counted in trading days:</p>
        <dl className="flex flex-col gap-xs">
          <Term name="Scalp">1 week (5 days) and 1 month (21 days). Liquidity floor $25M a day.</Term>
          <Term name="Swing">1 month (21 days) and 3 months (63 days). Liquidity floor $15M a day.</Term>
          <Term name="Long-term">6 months (126 days) and 1 year (252 days). Liquidity floor $50M a day.</Term>
        </dl>
        <p>
          For each window, every stock&apos;s return is turned into a <strong>z-score</strong>: how many standard deviations
          it sits from the average stock. A stock is an <strong>outlier</strong> when either z-score is above 2 or below −2.
        </p>
        <p>
          The <strong>score</strong> is the distance from the center of the group on both windows together: √(z₁² + z₂²).
          A higher score means a more unusual move. <strong>Direction</strong> is up or down from the window that crossed
          the line, or mixed when the two windows disagree.
        </p>
        <p>
          The scan runs every 30 minutes while the market is open and once after the close. The page checks for new data
          every 5 minutes.
        </p>
      </Section>

      <Section id="analysis" title="How a stock is analyzed">
        <p>
          The analysis uses about two years of daily closing prices. Returns are <strong>log returns</strong>, ln(today ÷
          yesterday). They add up over time and treat up and down moves the same way.
        </p>
        <dl className="flex flex-col gap-xs">
          <Term name="Expected value (EV)">
            The average result of following the sign of a forecast for one day: the win rate times the average win, minus the
            loss rate times the average loss. Positive EV is the minimum for any edge.
          </Term>
          <Term name="Sharpe ratio">
            Average daily return divided by its standard deviation, scaled to a year (× √252). It measures return per unit of
            risk.
          </Term>
          <Term name="Regime (AR(1))">
            A one-line model, next return = w × today&apos;s return + b. If w is below zero, moves tend to reverse
            (<strong>mean reversion</strong>). If w is above zero, moves tend to continue (<strong>momentum</strong>).
          </Term>
          <Term name="Models">
            Four forecasts of the next day&apos;s return: the AR(1) line, a small decision-tree model (XGBoost, depth 3) on
            weekday and recent-direction features, an online model that updates one day at a time
            (Passive-Aggressive), and a <strong>stacked</strong> model that blends the others with non-negative weights.
          </Term>
          <Term name="Signal strength">
            The stacked forecast for the next day, divided by one tenth of the stock&apos;s typical daily move, then passed
            through tanh. It runs from −1 to +1. A forecast worth a tenth of a normal day&apos;s move scores 0.76. Below 0.15
            either way counts as neutral.
          </Term>
          <Term name="Cost check">
            Every trade pays a round-trip cost (spread plus fees). The page compares the average size of the stacked
            forecast, per day, with an assumed cost of 10 basis points (0.10%). You can change the cost. If the forecast move
            is smaller than the cost, any edge is used up by trading costs.
          </Term>
        </dl>
        <p>
          <strong>No look-ahead.</strong> Every feature for day t uses data up to day t−1 only. Rolling averages are shifted
          by one day before they are computed. A unit test checks this.
        </p>
      </Section>

      <Section id="validation" title="How the models are checked">
        <ul className="flex list-disc flex-col gap-xxs pl-sm">
          <li>
            <strong>Time-ordered split.</strong> The oldest 75% of days train the models. The newest 25% test them. The days
            are never shuffled.
          </li>
          <li>
            <strong>Walk-forward.</strong> The models are re-fit many times, once with a growing (expanding) window and once
            with a fixed-length (rolling) window. Each fit is tested only on the days that come after it.
          </li>
          <li>
            <strong>Random baseline.</strong> The model&apos;s test return is compared with 1,000 strategies that pick up or
            down at random. If the model doesn&apos;t beat most of them, the page says there is no edge.
          </li>
          <li>
            <strong>Out-of-sample first.</strong> Headline numbers come from the test days only. Anything measured on the
            training days is labeled in-sample.
          </li>
        </ul>
      </Section>

      <Section id="limits" title="Limits of the data">
        <ul className="flex list-disc flex-col gap-xxs pl-sm">
          <li>
            Prices come from Yahoo Finance through an unofficial library. They can be delayed, missing or wrong, and there is
            no service guarantee.
          </li>
          <li>
            There is no order book data. Order-book imbalance and mid-price need level-2 quotes, which this source doesn&apos;t
            provide, so those measures are shown as unavailable rather than estimated.
          </li>
          <li>
            Daily bars only. The forecast horizon is one trading day. Intraday patterns are not measured.
          </li>
          <li>
            Z-scores use the plain mean and standard deviation. A few very large moves widen the spread, which makes
            smaller moves look less unusual.
          </li>
          <li>
            About 125 test days is a small sample. A good-looking result can still be luck. The page warns you when the sample
            is small or the model doesn&apos;t beat random.
          </li>
          <li>Past returns don&apos;t predict future returns reliably. Nothing here is a recommendation.</li>
        </ul>
      </Section>
    </article>
  );
}
