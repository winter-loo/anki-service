import adapter from '@sveltejs/adapter-static';

/** @type {import('@sveltejs/kit').Config} */
const config = {
	kit: {
		adapter: adapter({
			pages: 'build',
			assets: 'build',
			fallback: '404.html', // Enable SPA mode with a separate fallback file
			precompress: false,
			strict: true
		})
	}
};

export default config;
