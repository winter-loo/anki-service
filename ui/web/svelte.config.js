import adapter from '@sveltejs/adapter-static';

/** @type {import('@sveltejs/kit').Config} */
const config = {
	kit: {
		adapter: adapter({
			// Build static files into ui/out (kept out of git)
			pages: '../out',
			assets: '../out',
			fallback: '404.html', // Enable SPA mode with a separate fallback file
			precompress: false,
			strict: true
		})
	}
};

export default config;
