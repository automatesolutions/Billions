import localFont from 'next/font/local';

/** General Sans (Fontshare). Fetched by scripts/fetch-fonts.mjs; see DESIGN.md "Font substitution". */
export const generalSans = localFont({
  src: './fonts/GeneralSans-Variable.woff2',
  variable: '--font-general-sans',
  weight: '200 700',
  display: 'swap',
});
