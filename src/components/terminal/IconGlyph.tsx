import { ICONS } from "@/data/portfolio";

export function IconGlyph({ icon }: { icon: keyof typeof ICONS }) {
  return (
    <svg
      viewBox="0 0 24 24"
      width={16}
      height={16}
      fill="none"
      stroke="currentColor"
      strokeWidth={1.6}
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      {ICONS[icon].map((s, i) => {
        switch (s.t) {
          case "rect":
            return <rect key={i} x={s.x} y={s.y} width={s.w} height={s.h} rx={s.rx} />;
          case "circle":
            return <circle key={i} cx={s.cx} cy={s.cy} r={s.r} />;
          case "line":
            return <line key={i} x1={s.x1} y1={s.y1} x2={s.x2} y2={s.y2} />;
          case "polyline":
            return <polyline key={i} points={s.points} />;
          case "path":
            return <path key={i} d={s.d} />;
        }
      })}
    </svg>
  );
}
