<script>
  export let title = '';
  export let path = '';
  export let size = 20;
  // Use hex without #, or pass full CSS color
  export let color = 'currentColor';
  export let viewBox = '0 0 24 24';
</script>

<svg
  width={size}
  height={size}
  viewBox={viewBox}
  role="img"
  aria-label={title}
  xmlns="http://www.w3.org/2000/svg"
  style={`color:${color}`}
  fill="currentColor"
>
  <path d={path} />
</svg>
