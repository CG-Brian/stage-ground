import { Container } from "../ui/Container";
import { Sandbox } from "../sandbox/Sandbox";

export function SandboxSection() {
  return (
    <section className="py-6 sm:py-8">
      <Container wide>
        <Sandbox />
      </Container>
    </section>
  );
}
