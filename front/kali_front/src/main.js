import 'bootstrap/dist/css/bootstrap.css'
import './assets/main.css'

import PrimeVue from 'primevue/config'
import Aura from '@primevue/themes/aura'
import ToastService from 'primevue/toastservice'
import Toast from 'primevue/toast'
import ConfirmationService from 'primevue/confirmationservice'
import DialogService from 'primevue/dialogservice'

import { createApp } from 'vue'
import App from './App.vue'
import AppState from './plugins/appState.js'
import router from './router'

const app = createApp(App)

app.use(router)

app.use(PrimeVue, {
  theme: {
    preset: Aura,
    options: {
      prefix: 'p',
      darkModeSelector: '.p-dark',
      cssLayer: false,
    },
  },
})
app.use(AppState)
app.use(ConfirmationService)
app.use(ToastService)
app.use(DialogService)

app.use(ToastService)
app.component('Toast', Toast)

app.mount('#app')
