import { Nav } from "@/components/Nav";
import { Footer } from "@/components/Footer";
import { SandboxProvider } from "@/components/sandbox/sandbox-context";
import { Hero } from "@/components/sections/Hero";
import { SandboxSection } from "@/components/sections/SandboxSection";
import { WhyThisMatters } from "@/components/sections/WhyThisMatters";
import { M0Paradox } from "@/components/sections/M0Paradox";
import { AggregateResults } from "@/components/sections/AggregateResults";
import { InterventionLadder } from "@/components/sections/InterventionLadder";
import { Methodology } from "@/components/sections/Methodology";

export default function Home() {
  return (
    <SandboxProvider>
      <Nav />
      <main>
        <Hero />
        <SandboxSection />
        <WhyThisMatters />
        <M0Paradox />
        <AggregateResults />
        <InterventionLadder />
        <Methodology />
      </main>
      <Footer />
    </SandboxProvider>
  );
}
