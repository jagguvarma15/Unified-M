/**
 * SolidJS-compatible wrappers for lucide-react icons.
 *
 * lucide-react icons are React.forwardRef objects (not callable functions).
 * SolidJS's JSX transform calls components as functions, which crashes with
 * "X is not a function". Additionally, even if callable, they return React
 * element trees (nested forwardRefs), not real DOM nodes.
 *
 * This module wraps each icon so it:
 *   1. Is callable (a plain function)
 *   2. Returns a real SVG DOM node via react-dom/server's renderToStaticMarkup
 */

import type { JSX } from "solid-js";
import React from "react";
import { renderToStaticMarkup } from "react-dom/server";

// ---------------------------------------------------------------------------
// Icon wrapper
// ---------------------------------------------------------------------------

export interface IconProps {
  size?: number | string;
  class?: string;
  color?: string;
  strokeWidth?: number | string;
  "aria-hidden"?: boolean | "true" | "false";
  [key: string]: unknown;
}

export type SolidIcon = (props: IconProps) => JSX.Element;

/** Wrap a lucide-react forwardRef icon into a SolidJS-callable component. */
function wrap(reactIcon: any): SolidIcon {
  return ((props: IconProps): any => {
    const reactProps: Record<string, unknown> = { ...props };
    // SolidJS uses `class`, React uses `className`
    if (reactProps.class) {
      reactProps.className = reactProps.class;
      delete reactProps.class;
    }
    // Render the React component tree to an HTML string
    const el = React.createElement(reactIcon, reactProps);
    const html = renderToStaticMarkup(el);
    // Parse the HTML string into a real DOM node
    const template = document.createElement("template");
    template.innerHTML = html;
    return template.content.firstChild;
  }) as SolidIcon;
}

// ---------------------------------------------------------------------------
// Wrapped icon re-exports
// ---------------------------------------------------------------------------

import {
  Activity as _Activity,
  AlertCircle as _AlertCircle,
  AlertTriangle as _AlertTriangle,
  ArrowDown as _ArrowDown,
  ArrowUp as _ArrowUp,
  ArrowUpDown as _ArrowUpDown,
  BarChart2 as _BarChart2,
  BarChart3 as _BarChart3,
  Blocks as _Blocks,
  Brain as _Brain,
  Calculator as _Calculator,
  Check as _Check,
  CheckCircle as _CheckCircle,
  CheckCircle2 as _CheckCircle2,
  ChevronDown as _ChevronDown,
  Circle as _Circle,
  ClipboardCheck as _ClipboardCheck,
  Clock as _Clock,
  Cloud as _Cloud,
  Copy as _Copy,
  Cpu as _Cpu,
  Crosshair as _Crosshair,
  Database as _Database,
  DollarSign as _DollarSign,
  Download as _Download,
  ExternalLink as _ExternalLink,
  FileCheck as _FileCheck,
  FileText as _FileText,
  Gauge as _Gauge,
  GitCompareArrows as _GitCompareArrows,
  GitMerge as _GitMerge,
  HardDrive as _HardDrive,
  History as _History,
  Info as _Info,
  LayoutDashboard as _LayoutDashboard,
  Link2 as _Link2,
  Loader2 as _Loader2,
  Megaphone as _Megaphone,
  Minus as _Minus,
  Play as _Play,
  Plus as _Plus,
  Printer as _Printer,
  RefreshCw as _RefreshCw,
  Server as _Server,
  Settings as _Settings,
  Shield as _Shield,
  ShieldCheck as _ShieldCheck,
  Sparkles as _Sparkles,
  Stethoscope as _Stethoscope,
  Target as _Target,
  TestTube as _TestTube,
  Trash2 as _Trash2,
  TrendingDown as _TrendingDown,
  TrendingUp as _TrendingUp,
  Upload as _Upload,
  X as _X,
  XCircle as _XCircle,
  Zap as _Zap,
} from "lucide-react";

export const Activity = wrap(_Activity);
export const AlertCircle = wrap(_AlertCircle);
export const AlertTriangle = wrap(_AlertTriangle);
export const ArrowDown = wrap(_ArrowDown);
export const ArrowUp = wrap(_ArrowUp);
export const ArrowUpDown = wrap(_ArrowUpDown);
export const BarChart2 = wrap(_BarChart2);
export const BarChart3 = wrap(_BarChart3);
export const Blocks = wrap(_Blocks);
export const Brain = wrap(_Brain);
export const Calculator = wrap(_Calculator);
export const Check = wrap(_Check);
export const CheckCircle = wrap(_CheckCircle);
export const CheckCircle2 = wrap(_CheckCircle2);
export const ChevronDown = wrap(_ChevronDown);
export const Circle = wrap(_Circle);
export const ClipboardCheck = wrap(_ClipboardCheck);
export const Clock = wrap(_Clock);
export const Cloud = wrap(_Cloud);
export const Copy = wrap(_Copy);
export const Cpu = wrap(_Cpu);
export const Crosshair = wrap(_Crosshair);
export const Database = wrap(_Database);
export const DollarSign = wrap(_DollarSign);
export const Download = wrap(_Download);
export const ExternalLink = wrap(_ExternalLink);
export const FileCheck = wrap(_FileCheck);
export const FileText = wrap(_FileText);
export const Gauge = wrap(_Gauge);
export const GitCompareArrows = wrap(_GitCompareArrows);
export const GitMerge = wrap(_GitMerge);
export const HardDrive = wrap(_HardDrive);
export const History = wrap(_History);
export const Info = wrap(_Info);
export const LayoutDashboard = wrap(_LayoutDashboard);
export const Link2 = wrap(_Link2);
export const Loader2 = wrap(_Loader2);
export const Megaphone = wrap(_Megaphone);
export const Minus = wrap(_Minus);
export const Play = wrap(_Play);
export const Plus = wrap(_Plus);
export const Printer = wrap(_Printer);
export const RefreshCw = wrap(_RefreshCw);
export const Server = wrap(_Server);
export const Settings = wrap(_Settings);
export const Shield = wrap(_Shield);
export const ShieldCheck = wrap(_ShieldCheck);
export const Sparkles = wrap(_Sparkles);
export const Stethoscope = wrap(_Stethoscope);
export const Target = wrap(_Target);
export const TestTube = wrap(_TestTube);
export const Trash2 = wrap(_Trash2);
export const TrendingDown = wrap(_TrendingDown);
export const TrendingUp = wrap(_TrendingUp);
export const Upload = wrap(_Upload);
export const X = wrap(_X);
export const XCircle = wrap(_XCircle);
export const Zap = wrap(_Zap);

/** Type alias for use in interfaces that reference an icon component */
export type LucideIcon = SolidIcon;
