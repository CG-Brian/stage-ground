import { Nav } from "@/components/Nav";
import { Footer } from "@/components/Footer";
import { Hero } from "@/components/sections/Hero";
import { CoreProblem } from "@/components/sections/CoreProblem";
import { WhatWeMeasure } from "@/components/sections/WhatWeMeasure";
import { InterventionLadder } from "@/components/sections/InterventionLadder";
import { MainResults } from "@/components/sections/MainResults";
import { EvidenceBinding } from "@/components/sections/EvidenceBinding";
import { M0Paradox } from "@/components/sections/M0Paradox";
import { CaseExplorerSection } from "@/components/sections/CaseExplorerSection";
import { TNMDifferences } from "@/components/sections/TNMDifferences";
import { Methodology } from "@/components/sections/Methodology";

export default function Home() {
  return (
    <>
      <Nav />
      <main>
        <Hero />
        <CoreProblem />
        <WhatWeMeasure />
        <InterventionLadder />
        <MainResults />
        <EvidenceBinding />
        <M0Paradox />
        <CaseExplorerSection />
        <TNMDifferences />
        <Methodology />
      </main>
      <Footer />
    </>
  );
}
