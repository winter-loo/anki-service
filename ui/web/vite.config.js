import { sveltekit } from '@sveltejs/kit/vite';
import { defineConfig } from 'vite';
import tailwindcss from '@tailwindcss/vite';

export default defineConfig(({ mode }) => {
	const isRelease = mode === 'production' || mode === 'release';

	return {
		plugins: [sveltekit(), tailwindcss()],
		build: {
			minify: isRelease,
			sourcemap: !isRelease
		}
	};
});
