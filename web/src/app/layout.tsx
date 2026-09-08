import type { Metadata } from "next";
import { Inter, Source_Serif_4, JetBrains_Mono } from "next/font/google";
import "./globals.css";

const inter = Inter({
  variable: "--font-inter",
  subsets: ["latin"],
  display: "swap",
});

const sourceSerif = Source_Serif_4({
  variable: "--font-source-serif",
  subsets: ["latin"],
  display: "swap",
});

const jbMono = JetBrains_Mono({
  variable: "--font-jbmono",
  subsets: ["latin"],
  display: "swap",
});

export const metadata: Metadata = {
  title: "StageGround — Source-grounded evaluation of clinical LLM extraction",
  description:
    "StageGround evaluates whether LLM-generated TNM cancer staging labels are actually supported by the source pathology report, rather than merely matching a gold label.",
};

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html
      lang="en"
      className={`${inter.variable} ${sourceSerif.variable} ${jbMono.variable} h-full antialiased`}
    >
      <body className="min-h-full flex flex-col bg-background text-foreground">
        {children}
      </body>
    </html>
  );
}
