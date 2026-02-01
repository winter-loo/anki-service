// Minimal Supabase Auth bootstrap for anki-service UI.
// Loads config from /ui-config.json, initializes supabase-js, and exposes:
//   window.__auth = { supabase, getAccessToken }

export async function initAuth() {
  const cfgRes = await fetch('/ui-config.json', { cache: 'no-store' });
  if (!cfgRes.ok) throw new Error('Missing /ui-config.json (server not configured)');
  const cfg = await cfgRes.json();

  const supabaseUrl = cfg.supabaseUrl;
  const supabaseKey = cfg.supabasePublishableKey;
  if (!supabaseUrl || !supabaseKey) {
    throw new Error('Supabase config missing. Set PUBLIC_SUPABASE_URL and PUBLIC_SUPABASE_PUBLISHABLE_DEFAULT_KEY (or SUPABASE_PROJECT_URL/SUPABASE_PUBLISHABLE_KEY) on the server.');
  }

  const { createClient } = await import('https://esm.sh/@supabase/supabase-js@2');
  const supabase = createClient(supabaseUrl, supabaseKey);

  async function getAccessToken() {
    const { data, error } = await supabase.auth.getSession();
    if (error) throw error;
    const token = data?.session?.access_token;
    return token || null;
  }

  window.__auth = { supabase, getAccessToken, cfg };
  return window.__auth;
}
