<script>
  import { onMount } from 'svelte';
  import { createClient } from '@supabase/supabase-js';
  import { PUBLIC_SUPABASE_URL, PUBLIC_SUPABASE_PUBLISHABLE_KEY } from '$env/static/public';
  import Auth from '../components/Auth.svelte';
  import Home from '../components/Home.svelte';

  let loading = true;
  /** @type {import('@supabase/supabase-js').Session | null} */
  let session = null;
  /** @type {import('@supabase/supabase-js').SupabaseClient} */
  let supabase;
  let errorMsg = '';

  onMount(async () => {
    try {
      supabase = createClient(PUBLIC_SUPABASE_URL, PUBLIC_SUPABASE_PUBLISHABLE_KEY);

      // 2. Check session (now safe to call directly)
      const { data } = await supabase.auth.getSession();
      session = data.session;

      // Listen for auth changes
      supabase.auth.onAuthStateChange((_event, _session) => {
        session = _session;
      });

      // Handle password reset return
      const url = new URL(window.location.href);
      if (window.location.hash.includes('type=recovery') || url.searchParams.has('code')) {
        // Clear hash to prevent loops/leaks
        window.history.replaceState(null, '', window.location.pathname);
        alert('Please reset your password in settings (pending implementation).');
      }
    } catch (e) {
      errorMsg = e.message;
    } finally {
      loading = false;
    }
  });
</script>

{#if loading}
  <div class="min-h-screen flex items-center justify-center bg-white dark:bg-[#0F1419]">
      <div class="animate-spin rounded-full h-8 w-8 border-t-2 border-b-2 border-[#FF6B00]"></div>
  </div>
{:else if errorMsg}
  <div class="p-8 text-center text-red-500">
      <p>Error loading app: {errorMsg}</p>
  </div>
{:else if session}
  <Home {supabase} user={session.user} />
{:else}
  <Auth {supabase} on:authSuccess={() => { /* session updates via onAuthStateChange */ }} />
{/if}
