<template>
  <div ref="plotElement"></div>
</template>

<script>
import Plotly from 'plotly.js-dist'

export default {
  name: 'PlotlyChart',
  props: {
    data: Array,
    layout: Object,
  },
  mounted() {
    this.renderChart()
  },
  methods: {
    renderChart() {
      Plotly.newPlot(this.$refs.plotElement, this.data, this.layout, {
        displaylogo: false,
        responsive: true,
      })
    },

    reRenderChart() {
      console.log(this.data, this.layout)
      Plotly.react(this.$refs.plotElement, this.data, this.layout, {
        displaylogo: false,
        responsive: true,
      })
    },
  },
  watch: {
    data: 'renderChart',
  },
}
</script>
