import { Section } from "@/components/ui";
import OpticsPanel from "@/components/optics/OpticsPanel";

export default function OpticsPage() {
  return (
    <Section
      title="光学台"
      subtitle="TERS/THz 位移台手动控制 · 延迟线 · Wiggle 自检 · 泵浦-探测空跑"
    >
      <OpticsPanel />
    </Section>
  );
}
