<template>
  <form enctype="multipart/form-data" @submit.prevent="createUserHandler">
    <div class="mb-3 row">
      <label for="name" class="col-sm-2 col-form-label">Название вакансии</label>
      <div class="col-sm-10">
        <input type="text" class="form-control" id="name" name="name" />
      </div>
    </div>
    <div class="row mb-3">
      <label for="input_houses" class="col-sm-4 col-form-label">extraversion</label>
      <div class="col-sm-8">
        <label for="areaTreshold" class="form-label">{{ extraversion }}</label>
        <input
          type="range"
          v-model="extraversion"
          class="form-range"
          min="0"
          max="1"
          step="0.01"
          id="extraversion"
          name="extraversion"
          required
        />
      </div>
    </div>

    <div class="row mb-3">
      <label for="input_houses" class="col-sm-4 col-form-label">neuroticism</label>
      <div class="col-sm-8">
        <label for="areaTreshold" class="form-label">{{ neuroticism }}</label>
        <input
          type="range"
          v-model="neuroticism"
          class="form-range"
          min="0"
          max="1"
          step="0.01"
          id="neuroticism"
          name="neuroticism"
          required
        />
      </div>
    </div>

    <div class="row mb-3">
      <label for="input_houses" class="col-sm-4 col-form-label">agreeableness</label>
      <div class="col-sm-8">
        <label for="areaTreshold" class="form-label">{{ agreeableness }}</label>
        <input
          type="range"
          v-model="agreeableness"
          class="form-range"
          min="0"
          max="1"
          step="0.01"
          id="agreeableness"
          name="agreeableness"
          required
        />
      </div>
    </div>

    <div class="row mb-3">
      <label for="input_houses" class="col-sm-4 col-form-label">conscientiousness</label>
      <div class="col-sm-8">
        <label for="areaTreshold" class="form-label">{{ conscientiousness }}</label>
        <input
          type="range"
          v-model="conscientiousness"
          class="form-range"
          min="0"
          max="1"
          step="0.01"
          id="conscientiousness"
          name="conscientiousness"
          required
        />
      </div>
    </div>

    <div class="row mb-3">
      <label for="input_houses" class="col-sm-4 col-form-label">openness</label>
      <div class="col-sm-8">
        <label for="areaTreshold" class="form-label">{{ openness }}</label>
        <input
          type="range"
          v-model="openness"
          class="form-range"
          min="0"
          max="1"
          step="0.01"
          id="openness"
          name="openness"
          required
        />
      </div>
    </div>
    <button type="submit" class="btn btn-primary">Добавить</button>
  </form>
</template>

<script>
import { createVacancy } from '@/api'

export default {
  data() {
    return {
      extraversion: 0.5,
      neuroticism: 0.5,
      agreeableness: 0.5,
      conscientiousness: 0.5,
      openness: 0.5,
    }
  },
  emits: {
    vacancy_created: Object,
  },
  methods: {
    async createUserHandler(event) {
      let formData = new FormData(event.target)
      let response = await createVacancy(formData)
      this.$emit('vacancy_created', response)
    },
  },
}
</script>

<style></style>
