# Copyright 2021 The TF-Coder Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""Tests for filter_group.py."""

from absl.testing import absltest
from tf_coder import filter_group


class FilterGroupTest(absltest.TestCase):

  def test_filter_group_enum_names(self):
    for enum_value in filter_group.FilterGroup:
      self.assertEqual(enum_value.name, enum_value.value)


if __name__ == '__main__':
  absltest.main()
