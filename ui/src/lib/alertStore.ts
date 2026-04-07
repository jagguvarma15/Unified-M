import { createSignal } from "solid-js";

/** Global unacknowledged alert count — written by AlertsCenter, read by TopNavbar. */
export const [alertCount, setAlertCount] = createSignal(
  // Seed from demo data on first load (1 unacknowledged event)
  1,
);
