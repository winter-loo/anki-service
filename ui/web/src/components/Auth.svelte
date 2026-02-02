<script>
  import { createEventDispatcher } from 'svelte';
  import { BookOpen, Eye, EyeOff } from '@lucide/svelte';
  import { siWechat, siApple } from 'simple-icons/icons';
  import BrandIcon from './BrandIcon.svelte';
  
  export let supabase;
  
  let email = '';
  let password = '';
  let showPassword = false;
  let errorMsg = '';
  let loading = false;
  let view = 'login'; // login, signup, forgot

  const dispatch = createEventDispatcher();

  async function handleLogin() {
    if (!supabase) return;
    loading = true;
    errorMsg = '';
    const { error } = await supabase.auth.signInWithPassword({ email, password });
    loading = false;
    if (error) {
      errorMsg = error.message;
    } else {
      dispatch('authSuccess');
    }
  }

  async function handleSignup() {
      if (!supabase) return;
      loading = true;
      errorMsg = '';
      const redirectTo = window.location.origin;
      const { error } = await supabase.auth.signUp({ email, password, options: { emailRedirectTo: redirectTo } });
      loading = false;
      if (error) {
          errorMsg = error.message;
      } else {
          alert('Check your email for the confirmation link!');
          view = 'login';
      }
  }

  async function handleForgot() {
      if (!supabase) return;
      if (!email) {
          errorMsg = 'Please enter your email first';
          return;
      }
      loading = true;
      const redirectTo = window.location.origin;
      const { error } = await supabase.auth.resetPasswordForEmail(email, { redirectTo });
      loading = false;
      if (error) {
          errorMsg = error.message;
      } else {
          alert('Password reset email sent.');
          view = 'login';
      }
  }

  async function handleSocial(provider) {
      if (!supabase) return;
      const redirectTo = window.location.origin;
      const { error } = await supabase.auth.signInWithOAuth({
        provider,
        options: { redirectTo },
      });
      if (error) errorMsg = error.message;
  }
</script>

<div class="min-h-screen flex flex-col items-center justify-center p-6 font-display bg-background-light dark:bg-background-dark text-slate-900 dark:text-slate-100">
    <div class="w-full max-w-[400px] space-y-10">
        <div class="flex flex-col items-center space-y-6">
            <!-- Logo -->
            <div class="w-12 h-12 bg-white dark:bg-card-dark border border-slate-200 dark:border-slate-800 rounded-full flex items-center justify-center shadow-sm">
                <!-- Memit Logo (Book icon replacement) -->
                <BookOpen class="text-primary" size={24} strokeWidth={2.2} />
            </div>
            
            <!-- Header -->
            <div class="text-center space-y-2">
                <h1 class="text-2xl font-semibold tracking-tight text-slate-900 dark:text-white">
                    {#if view === 'login'}Log in to Memit
                    {:else if view === 'signup'}Create an account
                    {:else}Reset Password{/if}
                </h1>
                <p class="text-sm text-slate-500 dark:text-slate-400">
                    {#if view === 'login'}
                        Don't have an account? <button class="text-slate-900 dark:text-white font-medium hover:underline underline-offset-4" on:click={() => view = 'signup'}>Sign up</button>.
                    {:else if view === 'signup'}
                        Already have an account? <button class="text-slate-900 dark:text-white font-medium hover:underline underline-offset-4" on:click={() => view = 'login'}>Log in</button>.
                    {:else}
                        Remembered it? <button class="text-slate-900 dark:text-white font-medium hover:underline underline-offset-4" on:click={() => view = 'login'}>Log in</button>.
                    {/if}
                </p>
            </div>
        </div>

        {#if view !== 'forgot'}
        <!-- Social Buttons Grid -->
        <div class="grid grid-cols-3 gap-3">
            <button aria-label="Continue with Google" on:click={() => handleSocial('google')} class="flex items-center justify-center py-2.5 px-4 bg-white dark:bg-card-dark border border-slate-200 dark:border-slate-800 rounded-full hover:bg-slate-50 dark:hover:bg-slate-800 transition-all duration-200 shadow-sm">
                <svg class="w-5 h-5" viewBox="0 0 24 24"><path d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z" fill="#4285F4"></path><path d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z" fill="#34A853"></path><path d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l2.85-2.22.81-.62z" fill="#FBBC05"></path><path d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z" fill="#EA4335"></path></svg>
            </button>
            <button aria-label="Continue with Apple" class="flex items-center justify-center py-2.5 px-4 bg-white dark:bg-card-dark border border-slate-200 dark:border-slate-800 rounded-full hover:bg-slate-50 dark:hover:bg-slate-800 transition-all duration-200 shadow-sm">
                <BrandIcon title={siApple.title} path={siApple.path} size={20} color="currentColor" />
            </button>
            <button aria-label="Continue with WeChat" class="flex items-center justify-center py-2.5 px-4 bg-white dark:bg-card-dark border border-slate-200 dark:border-slate-800 rounded-full hover:bg-slate-50 dark:hover:bg-slate-800 transition-all duration-200 shadow-sm">
                <BrandIcon title={siWechat.title} path={siWechat.path} size={20} color="#07C160" />
            </button>
        </div>

        <!-- OR Divider -->
        <div class="relative">
            <div class="absolute inset-0 flex items-center">
                <span class="w-full border-t border-slate-200 dark:border-slate-800"></span>
            </div>
            <div class="relative flex justify-center text-[10px] uppercase font-bold text-slate-400">
                <span class="bg-background-light dark:bg-background-dark px-4 tracking-widest">or</span>
            </div>
        </div>
        {/if}

        <!-- Form -->
        <form class="space-y-6" on:submit|preventDefault={() => {
            if (view === 'login') handleLogin();
            else if (view === 'signup') handleSignup();
            else handleForgot();
        }}>
            {#if errorMsg}
                <div class="text-red-500 text-sm text-center bg-red-50 dark:bg-red-900/10 py-2 rounded-lg">{errorMsg}</div>
            {/if}

            <div class="space-y-2">
                <label for="email" class="text-sm font-semibold text-slate-900 dark:text-white">Email address</label>
                <input type="email" id="email" bind:value={email} required
                       class="w-full px-4 py-3 bg-white dark:bg-card-dark border border-slate-200 dark:border-slate-700 rounded-xl text-slate-900 dark:text-white placeholder:text-slate-400 focus:ring-2 focus:ring-primary/20 focus:border-primary outline-none transition-all" 
                       placeholder="name@example.com">
            </div>

            {#if view !== 'forgot'}
            <div class="space-y-2">
                <div class="flex items-center justify-between">
                    <label for="password" class="text-sm font-semibold text-slate-900 dark:text-white">Password</label>
                    {#if view === 'login'}
                    <button type="button" class="text-xs font-medium text-slate-400 hover:text-slate-900 dark:hover:text-white transition-colors" on:click={() => view = 'forgot'}>Forgot password?</button>
                    {/if}
                </div>
                <div class="relative">
                    <input type={showPassword ? 'text' : 'password'} id="password" bind:value={password} required
                           class="w-full px-4 py-3 bg-white dark:bg-card-dark border border-slate-200 dark:border-slate-700 rounded-xl text-slate-900 dark:text-white placeholder:text-slate-400 focus:ring-2 focus:ring-primary/20 focus:border-primary outline-none transition-all" 
                           placeholder="••••••••">
                    <button type="button" class="absolute right-3 top-1/2 -translate-y-1/2 text-slate-400 hover:text-slate-600 dark:hover:text-slate-200" on:click={() => showPassword = !showPassword}>
                        {#if showPassword}
                          <EyeOff size={18} />
                        {:else}
                          <Eye size={18} />
                        {/if}
                    </button>
                </div>
            </div>
            {/if}

            <button type="submit" disabled={loading}
                    class="w-full bg-primary text-white font-semibold py-3.5 rounded-xl hover:bg-primary-dark transition-all duration-200 shadow-sm active:scale-[0.98] shadow-primary/20">
                {#if loading}Processing...
                {:else if view === 'login'}Log In
                {:else if view === 'signup'}Sign Up
                {:else}Send Reset Link{/if}
            </button>
        </form>

        <!-- Terms Footer -->
        <p class="text-center text-[10px] text-slate-400 leading-relaxed max-w-[280px] mx-auto">
            By signing in, you agree to our 
            <a href="/terms" class="text-slate-500 underline underline-offset-2 hover:text-slate-900 dark:hover:text-white">Terms of Service</a> 
            and 
            <a href="/privacy" class="text-slate-500 underline underline-offset-2 hover:text-slate-900 dark:hover:text-white">Privacy Policy</a>.
        </p>
    </div>
</div>
