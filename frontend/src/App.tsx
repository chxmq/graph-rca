import { FiActivity, FiBookOpen, FiZap } from "react-icons/fi";
import { Layout } from "./components/Layout";
import { StepTabs } from "./components/StepTabs";
import { LogUploadPanel } from "./components/LogUploadPanel";
import { DocsUploadPanel } from "./components/DocsUploadPanel";
import { IncidentResolutionPanel } from "./components/IncidentResolutionPanel";

function App() {
  return (
    <Layout>
      <StepTabs
        tabs={[
          {
            id: "log",
            label: "1. Log analysis",
            icon: <FiActivity />,
            content: <LogUploadPanel />,
          },
          {
            id: "docs",
            label: "2. Documentation",
            icon: <FiBookOpen />,
            content: <DocsUploadPanel />,
          },
          {
            id: "resolve",
            label: "3. Incident resolution",
            icon: <FiZap />,
            content: <IncidentResolutionPanel />,
          },
        ]}
      />
    </Layout>
  );
}

export default App;

