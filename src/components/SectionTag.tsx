export function SectionTag({ children }: { children: string }) {
  return (
    <div className="mb-3 font-mono text-xs uppercase tracking-wider text-primary/80">
      <span className="text-muted-foreground">// </span>
      {children}
    </div>
  );
}

export function PageHeader({
  tag,
  title,
  description,
}: {
  tag: string;
  title: string;
  description?: string;
}) {
  return (
    <div className="animate-fade-in-up">
      <SectionTag>{tag}</SectionTag>
      <h1 className="text-4xl font-semibold tracking-tight sm:text-5xl">
        <span className="text-gradient">{title}</span>
      </h1>
      {description && (
        <p className="mt-4 max-w-2xl text-base text-muted-foreground sm:text-lg">
          {description}
        </p>
      )}
    </div>
  );
}
