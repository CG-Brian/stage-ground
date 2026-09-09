export function Container({
  children,
  className = "",
  wide = false,
}: {
  children: React.ReactNode;
  className?: string;
  /** Wider max-width for the sandbox's two-column layout. */
  wide?: boolean;
}) {
  return (
    <div
      className={`mx-auto w-full px-5 sm:px-8 ${wide ? "max-w-6xl" : "max-w-4xl"} ${className}`}
    >
      {children}
    </div>
  );
}
