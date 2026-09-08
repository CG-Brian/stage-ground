import { Container } from "./ui/Container";

export function Footer() {
  return (
    <footer className="mt-auto py-10">
      <Container className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-3 text-xs text-muted-2">
        <p>StageGround — a research/evaluation project, not a clinical product.</p>
        <a
          href="https://github.com/CG-Brian/stage-ground"
          target="_blank"
          rel="noreferrer"
          className="hover:text-foreground transition-colors"
        >
          github.com/CG-Brian/stage-ground
        </a>
      </Container>
    </footer>
  );
}
