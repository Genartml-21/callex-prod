// ─────────────────────────────────────────────────
//  CALLEX LOGO CONFIGURATION
//  Paste your logo URL below (PNG, SVG, or WebP).
//  This file is imported everywhere the logo appears.
// ─────────────────────────────────────────────────

export const LOGO_URL = 'https://iili.io/qF0blmQ.png';
// Example:
// export const LOGO_URL = 'https://your-cdn.com/callex-logo.png';
// export const LOGO_URL = '/logo.png';   ← if you drop logo.png into enterprise/frontend/public/

export const APP_NAME = 'Callex Enterprise';

// LogoImage component — renders your logo with an icon fallback
import { PhoneOutgoing } from 'lucide-react';

export function LogoImage({ size = 32, className = '' }) {
    if (!LOGO_URL || LOGO_URL === 'YOUR_LOGO_URL_HERE') {
        // Fallback: orange box with phone icon
        return (
            <div
                className={`rounded-xl bg-orange-500 flex items-center justify-center shrink-0 ${className}`}
                style={{ width: size, height: size }}
            >
                <PhoneOutgoing size={size * 0.5} className="text-white" />
            </div>
        );
    }
    return (
        <div
            className={`rounded-xl bg-orange-500 flex items-center justify-center shrink-0 p-1 ${className}`}
            style={{ width: size, height: size }}
        >
            <img
                src={LOGO_URL}
                alt={APP_NAME}
                className="w-full h-full object-contain"
                onError={(e) => { e.target.style.display = 'none'; }}
            />
        </div>
    );
}
