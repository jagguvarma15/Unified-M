import { BrowserRouter, Routes, Route } from "react-router-dom";
import Layout from "./components/Layout";
import Dashboard from "./pages/Dashboard";
import Contributions from "./pages/Contributions";
import Optimization from "./pages/Optimization";
import ResponseCurves from "./pages/ResponseCurves";
import Runs from "./pages/Runs";

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route element={<Layout />}>
          <Route index element={<Dashboard />} />
          <Route path="/contributions" element={<Contributions />} />
          <Route path="/optimization" element={<Optimization />} />
          <Route path="/curves" element={<ResponseCurves />} />
          <Route path="/runs" element={<Runs />} />
        </Route>
      </Routes>
    </BrowserRouter>
  );
}
