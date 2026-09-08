import { Container } from "./ui/Container";

const LINKS = [
  { href: "#problem", label: "The problem" },
  { href: "#measures", label: "Measures" },
  { href: "#results", label: "Results" },
  { href: "#m0-paradox", label: "M0 paradox" },
  { href: "#explorer", label: "Case explorer" },
  { href: "#methodology", label: "Methodology" },
];

const GITHUB_URL = "https://github.com/CG-Brian/stage-ground";

export function Nav() {
  return (
    <header className="sticky top-0 z-40 border-b border-border bg-background/85 backdrop-blur supports-[backdrop-filter]:bg-background/70">
      <Container className="flex h-14 items-center justify-between">
        <a
          href="#top"
          className="font-serif-display text-lg tracking-tight text-foreground"
        >
          StageGround
        </a>
        <nav
          aria-label="Section navigation"
          className="hidden md:flex items-center gap-6 text-sm text-muted"
        >
          {LINKS.map((l) => (
            <a
              key={l.href}
              href={l.href}
              className="hover:text-foreground focus-visible:text-foreground focus-visible:outline-none focus-visible:underline underline-offset-4 transition-colors"
            >
              {l.label}
            </a>
          ))}
        </nav>
        <a
          href={GITHUB_URL}
          target="_blank"
          rel="noreferrer"
          className="text-sm font-medium text-foreground border border-border-strong rounded-full px-3.5 py-1.5 hover:border-accent hover:text-accent transition-colors focus-visible:outline-2 focus-visible:outline-accent"
        >
          GitHub
        </a>
      </Container>
    </header>
  );
}
