import { useEffect, useState } from 'react';
import { api } from '../lib/api.js';
import { useStore } from '../store/index.js';
import { CheckCircle, Link, Unlink, ExternalLink } from 'lucide-react';

// Real integration logos using Clearbit Logo API + fallbacks
const INTEGRATIONS = [
    {
        slug: 'salesforce',
        name: 'Salesforce',
        description: 'CRM & customer data sync',
        logo: 'https://logo.clearbit.com/salesforce.com',
        color: '#00A1E0',
    },
    {
        slug: 'stripe',
        name: 'Stripe',
        description: 'Payment processing & billing',
        logo: 'https://logo.clearbit.com/stripe.com',
        color: '#635bff',
    },
    {
        slug: 'zapier',
        name: 'Zapier',
        description: 'Workflow automation',
        logo: 'https://logo.clearbit.com/zapier.com',
        color: '#FF4A00',
    },
    {
        slug: 'hubspot',
        name: 'HubSpot',
        description: 'Marketing & CRM platform',
        logo: 'https://logo.clearbit.com/hubspot.com',
        color: '#FF7A59',
    },
    {
        slug: 'zendesk',
        name: 'Zendesk',
        description: 'Customer support ticketing',
        logo: 'https://logo.clearbit.com/zendesk.com',
        color: '#03363D',
    },
    {
        slug: 'twilio',
        name: 'Twilio',
        description: 'Voice, SMS & telephony',
        logo: 'https://logo.clearbit.com/twilio.com',
        color: '#F22F46',
    },
    {
        slug: 'slack',
        name: 'Slack',
        description: 'Team notifications & alerts',
        logo: 'https://logo.clearbit.com/slack.com',
        color: '#4A154B',
    },
    {
        slug: 'segment',
        name: 'Segment',
        description: 'Customer data platform',
        logo: 'https://logo.clearbit.com/segment.com',
        color: '#52BD94',
    },
    {
        slug: 'intercom',
        name: 'Intercom',
        description: 'Customer messaging platform',
        logo: 'https://logo.clearbit.com/intercom.com',
        color: '#1F8AFF',
    },
];

function IntegrationLogo({ logo, name, color }) {
    const [imgError, setImgError] = useState(false);
    if (imgError) {
        return (
            <div
                className="w-12 h-12 rounded-2xl flex items-center justify-center text-white text-xl font-bold"
                style={{ backgroundColor: color }}
            >
                {name[0]}
            </div>
        );
    }
    return (
        <img
            src={logo}
            alt={name}
            className="w-12 h-12 rounded-2xl object-contain bg-white border border-gray-100 p-1.5 shadow-sm"
            onError={() => setImgError(true)}
        />
    );
}

export default function Integrations() {
    const [integrations, setIntegrations] = useState(INTEGRATIONS);
    const [busy, setBusy] = useState({});
    const { showToast } = useStore();

    useEffect(() => {
        api.integrations().then(dbIntegrations => {
            // Merge DB state (connected status) with our rich metadata
            const merged = INTEGRATIONS.map(meta => {
                const db = Array.isArray(dbIntegrations) ? dbIntegrations.find(d => d.slug === meta.slug) : null;
                return { ...meta, id: db?.id, connected: db?.connected ?? false };
            });
            setIntegrations(merged);
        }).catch(() => {
            // Fallback nicely if API is unvailable
            setIntegrations(INTEGRATIONS);
        });
    }, []);

    async function connect(int) {
        if (!int.id) return showToast('Integration DB record not found', 'error');
        setBusy(b => ({ ...b, [int.slug]: true }));
        await api.connectIntegration(int.id, { connectedAt: new Date().toISOString() });
        setIntegrations(ints => ints.map(i => i.slug === int.slug ? { ...i, connected: true } : i));
        showToast(`${int.name} connected successfully!`, 'success');
        setBusy(b => ({ ...b, [int.slug]: false }));
    }

    async function disconnect(int) {
        if (!int.id) return;
        setBusy(b => ({ ...b, [int.slug]: true }));
        await api.disconnectIntegration(int.id);
        setIntegrations(ints => ints.map(i => i.slug === int.slug ? { ...i, connected: false } : i));
        showToast(`${int.name} disconnected`, 'info');
        setBusy(b => ({ ...b, [int.slug]: false }));
    }

    const connectedCount = integrations.filter(i => i.connected).length;

    return (
        <div className="space-y-6">
            <div className="page-header">
                <div>
                    <h1 className="text-2xl font-bold text-gray-900">Integrations Marketplace</h1>
                    <p className="text-sm text-gray-400 mt-0.5">Connect your enterprise stack to the Callex voice platform</p>
                </div>
                <div className="flex items-center gap-3">
                    <div className="px-3 py-1.5 bg-orange-50 rounded-xl">
                        <span className="text-sm font-semibold text-orange-600">{connectedCount}/{integrations.length} Connected</span>
                    </div>
                </div>
            </div>

            <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4">
                {integrations.map(int => (
                    <div
                        key={int.slug}
                        className={`bg-white rounded-2xl border-2 p-5 hover:shadow-md transition-all duration-200 flex flex-col gap-4 ${int.connected ? 'border-orange-200 shadow-sm shadow-orange-50' : 'border-gray-100 hover:border-gray-200'
                            }`}
                    >
                        <div className="flex items-start justify-between">
                            <IntegrationLogo logo={int.logo} name={int.name} color={int.color} />
                            {int.connected && (
                                <span className="badge-green"><CheckCircle size={11} className="mr-1" />Active</span>
                            )}
                        </div>

                        <div className="flex-1">
                            <div className="font-semibold text-gray-900 text-sm mb-0.5">{int.name}</div>
                            <div className="text-xs text-gray-400 leading-relaxed">{int.description}</div>
                        </div>

                        <button
                            onClick={() => int.connected ? disconnect(int) : connect(int)}
                            disabled={busy[int.slug]}
                            className={`w-full text-sm font-semibold py-2 rounded-xl transition-all border flex items-center justify-center gap-1.5 ${int.connected
                                ? 'border-gray-200 text-gray-500 hover:border-red-200 hover:text-red-500 hover:bg-red-50'
                                : 'border-orange-200 text-orange-600 bg-orange-50 hover:bg-orange-500 hover:text-white'
                                }`}
                        >
                            {busy[int.slug] ? (
                                <span className="animate-spin text-gray-400">⟳</span>
                            ) : int.connected ? (
                                <><Unlink size={13} /> Disconnect</>
                            ) : (
                                <><Link size={13} /> Connect</>
                            )}
                        </button>
                    </div>
                ))}
            </div>
        </div>
    );
}
