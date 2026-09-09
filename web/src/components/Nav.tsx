import { Container } from "./ui/Container";

const LINKS = [
  { href: "#sandbox", label: "Sandbox" },
  { href: "#why", label: "Why it matters" },
  { href: "#m0-paradox", label: "M0 paradox" },
  { href: "#results", label: "Results" },
  { href: "#methodology", label: "Methodology" },
];

const GITHUB_URL = "https://github.com/CG-Brian/stage-ground";

export function Nav() {
  return (
    <header className="sticky top-0 z-40 border-b border-border bg-background/90 backdrop-blur supports-[backdrop-filter]:bg-background/75">
      <Container wide className="flex h-12 items-center justify-between">
        <a
          href="#top"
          className="font-mono-data text-sm font-semibold tracking-tight text-foreground"
        >
          StageGround
        </a>
        <nav
          aria-label="Section navigation"
          className="hidden md:flex items-center gap-5 text-[13px] text-muted"
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
          className="text-[13px] font-medium text-foreground border border-border-strong rounded px-3 py-1 hover:border-accent hover:text-accent transition-colors focus-visible:outline-2 focus-visible:outline-accent"
        >
          GitHub
        </a>
      </Container>
    </header>
  );
}
