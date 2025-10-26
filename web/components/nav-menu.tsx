'use client';

import Link from "next/link";
import { usePathname } from "next/navigation";
import { Button } from "@/components/ui/button";

interface NavItem {
  href: string;
  label: string;
}

const navItems: NavItem[] = [
  { href: "/dashboard", label: "Dashboard" },
  { href: "/outliers", label: "Outliers" },
  { href: "/portfolio", label: "Portfolio" },
  { href: "/trading/hft", label: "HFT Trading" },
  { href: "/trading/quantitative", label: "Quant Trading" },
];

export function NavMenu() {
  const pathname = usePathname();

  return (
    <nav className="flex gap-2">
      {navItems.map((item) => (
        <Link key={item.href} href={item.href}>
          <Button
            variant={pathname === item.href ? "default" : "ghost"}
            size="sm"
          >
            {item.label}
          </Button>
        </Link>
      ))}
    </nav>
  );
}

